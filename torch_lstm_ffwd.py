# Import the libraries. #
import torch
import torch.nn.functional as F

# LSTM Layer. #
class LSTMLayer(torch.nn.Module):
    def __init__(
        self, d_model, res_conn=False, rate=0.1):
        super(LSTMLayer, self).__init__()
        self.rate = rate
        self.d_model  = d_model
        self.res_conn = res_conn

        self.W = torch.nn.Linear(
            d_model, 4*d_model, bias=False)
        self.U = torch.nn.Linear(
            d_model, 4*d_model, bias=False)
        self.b = torch.nn.parameter.Parameter(
            torch.zeros(4*d_model))
        
        self.h_proj = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.ffwd_1 = torch.nn.Linear(
            d_model, 4*d_model, bias=False)
        self.ffwd_2 = torch.nn.Linear(
            4*d_model, d_model, bias=False)
        
        self.lnorm_1 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_2 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.dropout = torch.nn.Dropout(rate)
    
    def forward(
        self, x, c_prev, h_prev, training=True):
        # Layer Normalization. #
        x_norm = self.lnorm_1(x)

        # LSTM internal gates and states. #
        lin_proj = self.W(x_norm) + self.U(h_prev) + self.b
        c_tilde  = torch.tanh(lin_proj[:, (3*self.d_model):])
        tmp_gate = torch.sigmoid(lin_proj[:, :(3*self.d_model)])
        
        i_gate = tmp_gate[:, :self.d_model]
        o_gate = tmp_gate[:, (2*self.d_model):]
        f_gate = tmp_gate[:, self.d_model:(2*self.d_model)]
        
        c_next = torch.add(
            torch.multiply(f_gate, c_prev), 
            torch.multiply(i_gate, c_tilde))
        if training:
            c_next = self.dropout(c_next)
        
        h_next = torch.multiply(o_gate, torch.tanh(c_next))
        if training:
            h_next = self.dropout(h_next)
        
        # Feed forward. #
        y_ffwd = self.ffwd_2(
            F.relu(self.ffwd_1(self.lnorm_2(x))))

        # Residual Connection. #
        layer_output = y_ffwd + self.h_proj(h_next)
        if self.res_conn:
            layer_output = layer_output + x
        return c_next, h_next, layer_output

class LSTMNetwork(torch.nn.Module):
    def __init__(
        self, n_layers, d_model, vocab_size, 
        max_seq_length, rate=0.1, res_conn=False):
        super(LSTMNetwork, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.d_model  = d_model
        self.seq_len  = max_seq_length
        self.vocab_size = vocab_size
        
        # Embedding Layers. #
        self.dec_embed = torch.nn.Embedding(vocab_size, d_model)
        
        # LSTM Layers. #
        self.lstm_layers = torch.nn.ModuleList()
        for n_layer in range(n_layers):
            self.lstm_layers.append(LSTMLayer(
                d_model, rate=rate, res_conn=res_conn))
        self.emb_dropout = torch.nn.Dropout(rate)
    
    def forward(
        self, x, c_prev=None, h_prev=None, training=True):
        batch_sz = list(torch._shape_as_tensor(x))[0]
        zero_shp = [self.n_layers, batch_sz, self.d_model]
        if c_prev is None:
            c_prev = torch.zeros(
                zero_shp, dtype=torch.float32)
        
        if h_prev is None:
            h_prev = torch.zeros(
                zero_shp, dtype=torch.float32)
        
        if torch.cuda.is_available():
            c_prev = c_prev.to("cuda")
            h_prev = h_prev.to("cuda")
        
        x_tok_embed = self.dec_embed(x)
        if training:
            x_tok_embed = self.emb_dropout(x_tok_embed)
        
        c_next = []
        h_next = []

        layer_input = x_tok_embed
        for m in range(self.n_layers):
            output_tuple = self.lstm_layers[m](
                layer_input, c_prev[m, :, :], 
                h_prev[m, :, :], training=training)
            
            layer_input = output_tuple[2]
            c_next.append(torch.unsqueeze(output_tuple[0], 0))
            h_next.append(torch.unsqueeze(output_tuple[1], 0))
        
        c_next = torch.cat(c_next, dim=0)
        h_next = torch.cat(h_next, dim=0)
        lstm_output = output_tuple[2]
        return c_next, h_next, lstm_output

class LSTM(torch.nn.Module):
    def __init__(
        self, n_layers, d_model, vocab_size, 
        max_seq_length, rate=0.1, res_conn=False):
        super(LSTM, self).__init__()
        self.rate = rate
        self.n_layers = n_layers
        self.d_model  = d_model
        self.seq_len  = max_seq_length

        self.vocab_size = vocab_size
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(
            reduction="none")
        
        # Vocabulary index. #
        vocab_idx = torch.arange(
            0, vocab_size, dtype=torch.long)
        if torch.cuda.is_available():
            self.vocab_idx = vocab_idx.to("cuda")
        else:
            self.vocab_idx = vocab_idx
        del vocab_idx
        
        # Output projection. #
        self.lstm_model = LSTMNetwork(
            n_layers, d_model, vocab_size, 
            max_seq_length, rate=rate, res_conn=res_conn)
    
    def forward(
        self, x, c_prev=None, 
        h_prev=None, training=True):
        dec_logits = []
        seq_length = list(torch._shape_as_tensor(x))[1]

        for tm_idx in range(seq_length):
            output_tuple = self.lstm_model(
                x[:, tm_idx], c_prev=c_prev, 
                h_prev=h_prev, training=training)
            
            c_next = output_tuple[0]
            h_next = output_tuple[1]

            # Extract the logits. #
            param_emb = self.lstm_model.dec_embed(self.vocab_idx)
            tmp_logit = torch.matmul(
                output_tuple[2], torch.transpose(param_emb, 0, 1))
            dec_logits.append(torch.unsqueeze(tmp_logit, 1))
        return c_next, h_next, torch.cat(dec_logits, dim=1)
    
    def compute_ce_loss(
        self, x_input, x_output, seg_len=None):
        if seg_len is None:
            seg_len = self.seq_len
        
        if self.seq_len <= seg_len:
            n_segments = 1
        elif self.seq_len % seg_len == 0:
            n_segments = int(self.seq_len / seg_len)
        else:
            n_segments = int(self.seq_len / seg_len) + 1

        seq_ce_loss = 0.0
        tmp_lstm_tuple = self.forward(
            x_input, training=True)
        for n_segment in range(n_segments):
            l_st = n_segment * seg_len
            if n_segment != (n_segments-1):
                l_en = (n_segment+1) * seg_len
            else:
                l_en = self.seq_len
            tmp_dec_out = tmp_lstm_tuple[2]
            
            tmp_labels = x_output[:, l_st:l_en]
            tmp_logits = tmp_dec_out[:, l_st:l_en, :]
            seg_ce_loss = torch.sum(torch.sum(self.ce_loss_fn(
                torch.transpose(tmp_logits, 1, 2), tmp_labels), 1))
            seq_ce_loss += seg_ce_loss
        return seq_ce_loss
    
    def infer(self, x, k=1):
        input_len = list(torch._shape_as_tensor(x))[1]
        infer_ids = [torch.unsqueeze(x[:, 0], 1)]
        
        c_prev = None
        h_prev = None
        for step in range(self.seq_len):
            tmp_inputs = torch.cat(infer_ids, dim=1)
            with torch.no_grad():
                curr_input = torch.unsqueeze(
                    tmp_inputs[:, -1], dim=1)
                
                tmp_tuple = self.forward(
                    curr_input, c_prev=c_prev, 
                    h_prev=h_prev, training=False)
                tmp_logit = torch.squeeze(tmp_tuple[2], dim=1)

                c_prev = tmp_tuple[0]
                h_prev = tmp_tuple[1]

                if step < (input_len-1):
                    tmp_argmax = x[:, step+1]
                    infer_ids.append(torch.unsqueeze(tmp_argmax, 1))
                else:
                    if k == 1:
                        tmp_argmax = torch.argmax(
                            tmp_logit, dim=1)
                        infer_ids.append(torch.unsqueeze(tmp_argmax, 1))
                    else:
                        tmp_prob  = torch.softmax(tmp_logit, dim=1)

                        tmp_top_k  = torch.topk(tmp_prob, k=k)
                        tmp_sample = torch.multinomial(
                            tmp_top_k.values, 1)
                        tmp_index  = torch.gather(
                            tmp_top_k.indices, 1, tmp_sample)
                        infer_ids.append(tmp_index)
        return torch.cat(infer_ids, dim=1)
