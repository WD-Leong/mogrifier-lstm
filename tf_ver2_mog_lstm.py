
import tensorflow as tf

# LSTM Layer. #
class MogrifierLSTMLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, n_rounds=5, rate=0.1):
        super(MogrifierLSTMLayer, self).__init__()
        self.rate = rate
        self.n_rounds = n_rounds
        self.hidden_units = hidden_units

        # LSTM weights. #
        self.W = tf.keras.layers.Dense(4*hidden_units)
        self.U = tf.keras.layers.Dense(4*hidden_units)
        
        # LSTM bias. #
        self.b = self.add_weight(
            name="lstm_bias", shape=(4*hidden_units), 
            initializer="zeros", trainable=True)
        
        # Mogrifier weights. #
        self.W_mog = [tf.keras.layers.Dense(
            hidden_units) for _ in range(n_rounds)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    @tf.function
    def call(
        self, x_curr, c_prev, h_prev, training=True):
        x_mog_prev = x_curr
        h_mog_prev = h_prev

        # Mogrifier Layer before LSTM Layer. #
        for n_rd in range(self.n_rounds):
            if n_rd % 2 == 0:
                h_prev = tf.multiply(
                    h_mog_prev, 2*tf.nn.sigmoid(
                        self.W_mog[n_rd](x_mog_prev)))
                h_mog_prev = h_prev
            else:
                x_prev = tf.multiply(
                    x_mog_prev, 2*tf.nn.sigmoid(
                        self.W_mog[n_rd](h_mog_prev)))
                x_mog_prev = x_prev
        
        # Assign the LSTM inputs and states. #
        x_curr = x_mog_prev
        h_prev = h_mog_prev

        lin_proj = self.W(x_curr) + self.U(h_prev) + self.b
        gate_out = tf.split(tf.nn.sigmoid(
            lin_proj[:, :(3*self.hidden_units)]), 3, axis=-1)

        # LSTM Layer after Mogrifier Layer. #
        c_next = tf.add(
            tf.multiply(gate_out[1], c_prev), 
            tf.multiply(gate_out[0], tf.nn.tanh(
                lin_proj[:, (3*self.hidden_units):])))
        h_next = tf.multiply(gate_out[2], tf.nn.tanh(c_next))
        h_next = self.dropout(h_next, training=training)
        return (c_next, h_next)

class MogrifierLSTMNetwork(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, hidden_units, 
        n_rounds=5, rate=0.1, res_conn=True):
        super(MogrifierLSTMNetwork, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.n_rounds = n_rounds
        self.hidden_units = hidden_units
        
        # Layer Normalization. #
        self.norm_layers = [
            tf.keras.layers.LayerNormalization(
                epsilon=1.0e-6) for _ in range(n_layers)]
        
        # Decoder Layers. #
        self.dec_layers = [
            MogrifierLSTMLayer(
                hidden_units, rate=rate, 
                n_rounds=self.n_rounds) for _ in range(n_layers)]
    
    def call(self, x_input, c_prev, h_prev, training=True):
        c_next = []
        h_next = []

        prev_input  = x_input
        layer_input = x_input
        for m in range(self.n_layers):
            x_norm = self.norm_layers[m](layer_input)

            # Mogrifier LSTM Layer. #
            output_tuple = self.dec_layers[m](
                x_norm, c_prev[m], h_prev[m], training=training)
            
            c_next.append(
                tf.expand_dims(output_tuple[0], axis=0))
            h_next.append(
                tf.expand_dims(output_tuple[1], axis=0))

            # Residual Connection. #
            res_output = tf.add(
                layer_input, output_tuple[1])
            if self.res_conn:
                res_output += prev_input
            
            prev_input  = layer_input
            layer_input = res_output
        
        lstm_output = res_output
        c_next_stacked = tf.concat(c_next, axis=0)
        h_next_stacked = tf.concat(h_next, axis=0)
        return (c_next_stacked, h_next_stacked, lstm_output)

class LSTM(tf.keras.Model):
    def __init__(
        self, n_layers, hidden_units, 
        vocab_size, max_seq_length, 
        n_rounds=5, rate=0.1, res_conn=True):
        super(LSTM, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.n_rounds = n_rounds
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_length
        self.hidden_units = hidden_units

        # Vocabulary Embedding. #
        self.dec_embed = tf.keras.layers.Embedding(
            vocab_size, hidden_units, name="vocab_embedding")

        # Mogrifier LSTM Network. #
        self.lstm_model = MogrifierLSTMNetwork(
            n_layers, hidden_units, rate=rate, 
            n_rounds=n_rounds, res_conn=res_conn)
    
    def call(self, x, c_prev, h_prev, training=True):
        x_tok_embed  = self.dec_embed(x)
        output_tuple = self.lstm_model(
            x_tok_embed, c_prev, h_prev, training=training)
        
        c_next = output_tuple[0]
        h_next = output_tuple[1]

        # Get the embedding matrix. #
        x_vocab = tf.range(self.vocab_size)
        w_embed = self.dec_embed(x_vocab)

        # Return the vocab logits. #
        dec_logit = tf.matmul(
            output_tuple[2], w_embed, transpose_b=True)
        return c_next, h_next, dec_logit
    
    def decode(self, x, training=True):
        input_shape = x.shape
        batch_size  = input_shape[0]
        dec_length  = input_shape[1]
        zero_shape  = [
            self.n_layers, batch_size, self.hidden_units]
        
        # Initialise the states. #
        c_prev = tf.zeros(zero_shape, dtype=tf.float32)
        h_prev = tf.zeros(zero_shape, dtype=tf.float32)

        dec_logits = []
        for t_index in range(dec_length):
            next_tuple = self.call(
                x[:, t_index], c_prev, 
                h_prev, training=training)
            
            # Update the states. #
            c_prev = next_tuple[0]
            h_prev = next_tuple[1]
            
            # Append the output logits. #
            dec_logits.append(
                tf.expand_dims(next_tuple[2], axis=1))
        
        # Concatenate the output logits. #
        dec_logits = tf.concat(dec_logits, axis=1)
        return dec_logits
    
    def infer(self, x, gen_len=None, sample=False):
        input_len = x.shape[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        if gen_len is None:
            gen_len = self.max_seq_len
        
        batch_size = tf.shape(x)[0]
        zero_shape = [
            self.n_layers, batch_size, self.hidden_units]
        
        # Initialise the states. #
        c_prev = tf.zeros(zero_shape, dtype=tf.float32)
        h_prev = tf.zeros(zero_shape, dtype=tf.float32)

        for step in range(gen_len):
            curr_inputs = tf.concat(infer_ids, axis=1)
            
            next_tuple = self.call(
                curr_inputs[:, -1], 
                c_prev, h_prev, training=False)
            
            tmp_logit = next_tuple[2]
            if sample:
                tmp_sample = tf.random.categorical(
                    tmp_logit, 1, dtype=tf.int32)[:, 0]
            else:
                tmp_sample = tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32)
            
            # Update the states. #
            c_prev = next_tuple[0]
            h_prev = next_tuple[1]
            
            if step < (input_len-1):
                tmp_index = x[:, step+1]
            else:
                tmp_index = tmp_sample
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
