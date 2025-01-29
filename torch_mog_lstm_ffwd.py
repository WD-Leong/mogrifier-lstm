# Import the libraries. #
import torch
import torch.nn.functional as F

# LSTM Layer. #
class LSTMLayer(torch.nn.Module):
    def __init__(
        self, d_model, n_rounds=3, res_conn=False, rate=0.1):
        super(LSTMLayer, self).__init__()
        self.rate = rate
        self.d_model  = d_model
        self.res_conn = res_conn
        self.n_rounds = n_rounds

        self.W = torch.nn.Linear(
            d_model, 4*d_model, bias=False)
        self.U = torch.nn.Linear(
            d_model, 4*d_model, bias=False)
        self.b = torch.nn.parameter.Parameter(
            torch.zeros(4*d_model))
        
        self.W_mog = torch.nn.ModuleList()
        for n_round in range(n_rounds):
            self.W_mog.append(torch.nn.Linear(
                d_model, d_model, bias=False))
        
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

        # Mogrifier Layer. #
        x_mog_prev = x_norm
        h_mog_prev = h_prev

        # Mogrifier Layer before LSTM Layer. #
        for n_rd in range(self.n_rounds):
            if n_rd % 2 == 0:
                h_prev = torch.multiply(
                    h_mog_prev, 2*torch.sigmoid(
                        self.W_mog[n_rd](x_mog_prev)))
                h_mog_prev = h_prev
            else:
                x_prev = torch.multiply(
                    x_mog_prev, 2*torch.sigmoid(
                        self.W_mog[n_rd](h_mog_prev)))
                x_mog_prev = x_prev

        # LSTM internal gates and states. #
        lin_proj = torch.add(
            self.W(x_mog_prev), self.U(h_mog_prev) + self.b)
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
        max_seq_length, n_rounds=3, rate=0.1, res_conn=False):
        super(LSTMNetwork, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.d_model  = d_model
        self.n_rounds = n_rounds
        self.seq_len  = max_seq_length
        self.vocab_size = vocab_size
        
        # Embedding Layers. #
        self.dec_embed = torch.nn.Embedding(vocab_size, d_model)
        
        # LSTM Layers. #
        self.lstm_layers = torch.nn.ModuleList()
        for n_layer in range(n_layers):
            self.lstm_layers.append(LSTMLayer(
                d_model, n_rounds=n_rounds, 
                rate=rate, res_conn=res_conn))
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
        max_seq_length, n_rounds=3, rate=0.1, res_conn=False):
        super(LSTM, self).__init__()
        self.rate = rate
        self.n_layers = n_layers
        self.d_model  = d_model
        self.n_rounds = n_rounds
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
        
        # LSTM Model. #
        self.lstm_model = LSTMNetwork(
            n_layers, d_model, vocab_size, max_seq_length, 
            n_rounds=n_rounds, rate=rate, res_conn=res_conn)
    
    def forward(
        self, x, c_prev=None, 
        h_prev=None, training=True):
        output_tuple = self.lstm_model(
            x, c_prev=c_prev, h_prev=h_prev, training=training)
        
        c_next = output_tuple[0]
        h_next = output_tuple[1]

        # Extract the logits. #
        param_emb = self.lstm_model.dec_embed(self.vocab_idx)
        dec_logit = torch.matmul(
            output_tuple[2], torch.transpose(param_emb, 0, 1))
        return c_next, h_next, dec_logit
    
    def compute_ce_loss(
        self, x_input, x_output, c_init=None):
        # Initialise the LSTM states. #
        h_prev = None
        c_prev = c_init
        
        raw_ce_loss = []
        for step in range(self.seq_len):
            tmp_lstm_tuple = self.forward(
                x_input[:, step], c_prev=c_prev, 
                h_prev=h_prev, training=True)
            
            c_prev = tmp_lstm_tuple[0]
            h_prev = tmp_lstm_tuple[1]
            dec_logit = tmp_lstm_tuple[2]
            
            tmp_ce_loss = self.ce_loss_fn(
                dec_logit, x_output[:, step])
            raw_ce_loss.append(
                torch.unsqueeze(tmp_ce_loss, 1))
        
        seq_ce_loss = torch.cat(raw_ce_loss, dim=1)
        tot_ce_loss = torch.sum(torch.sum(seq_ce_loss, 1))
        return tot_ce_loss
    
    def infer(self, x, k=1, c_init=None):
        input_len = list(torch._shape_as_tensor(x))[1]
        infer_ids = [torch.unsqueeze(x[:, 0], 1)]
        
        with torch.no_grad():
            h_prev = None
            c_prev = c_init
            
            for step in range(self.seq_len):
                tmp_inputs = torch.cat(infer_ids, dim=1)
                curr_input = tmp_inputs[:, -1]
                
                tmp_tuple = self.forward(
                    curr_input, c_prev=c_prev, 
                    h_prev=h_prev, training=False)

                c_prev = tmp_tuple[0]
                h_prev = tmp_tuple[1]
                tmp_logit = tmp_tuple[2]
                
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
