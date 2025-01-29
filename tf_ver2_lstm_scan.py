# Import the libraries. #
import numpy as np
import tensorflow as tf

# Layer Normalization. #
class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, d_model, epsilon=1.0e-3, center=True):
        # center = True will return Layer Normalization, # 
        # center = False will return RMS Normalization.  #
        super(LayerNorm, self).__init__()
        self.center  = center
        self.epsilon = epsilon
        
        if center:
            self.beta = self.add_weight(
                name="beta", shape=d_model, 
                initializer="zeros", trainable=True)
        else:
            self.beta = 0.0
        
        self.gamma = self.add_weight(
            name="gamma", shape=d_model, 
            initializer="ones", trainable=True)
    
    def call(self, x):
        if self.center:
            x_mean  = tf.reduce_mean(x, axis=-1, keepdims=True)
            x_sigma = tf.math.sqrt(tf.reduce_mean(
                tf.square(x - x_mean), axis=-1, keepdims=True))
            
            x_scale = tf.divide(
                x - x_mean, x_sigma + self.epsilon)
        else:
            x_sigma = tf.math.sqrt(tf.reduce_mean(
                tf.square(x), axis=-1, keepdims=True))
            x_scale = tf.divide(x, x_sigma + self.epsilon)
        
        x_output = self.gamma * x_scale + self.beta
        return x_output

# LSTM Layer. #
class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, rate=0.1):
        super(LSTMLayer, self).__init__()
        self.rate = rate
        self.hidden_units = hidden_units
        
        # Linear projection layers. #
        self.linear = tf.keras.layers.Dense(4*hidden_units)
        
        # Layer Normalization. #
        self.dropout = tf.keras.layers.Dropout(rate)
        self.lx_norm = tf.keras.layers.LayerNormalization(epsilon=1.0e-6)
    
    def call(
        self, x_curr, c_prev, h_prev, training=True):
        x_norm = self.lx_norm(x_curr)
        
        batch_size  = tf.shape(x_curr)[0]
        lstm_input  = tf.concat([x_norm, h_prev], axis=1)
        shp_outputs = [batch_size, 4, self.hidden_units]
        tmp_outputs = tf.reshape(
            self.linear(lstm_input), shp_outputs)
        
        input_gate  = tf.nn.sigmoid(tmp_outputs[:, 0, :])
        forget_gate = tf.nn.sigmoid(tmp_outputs[:, 1, :])
        output_gate = tf.nn.sigmoid(tmp_outputs[:, 2, :])
        
        c_next = tf.add(
            tf.multiply(forget_gate, c_prev), tf.multiply(
                input_gate, tf.nn.tanh(tmp_outputs[:, 3, :])))
        c_next = self.dropout(c_next, training=training)
        h_next = tf.multiply(output_gate, tf.nn.tanh(c_next))
        h_next = self.dropout(h_next, training=training)
        return (c_next, h_next)

# For Decoder. #
class LSTMDecoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, hidden_units, 
        alpha=0.1, res_conn=True, rate=0.1):
        super(LSTMDecoder, self).__init__()
        
        self.rate  = rate
        self.alpha = alpha
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.hidden_units = hidden_units
        
        # Decoder Layers. #
        self.lstm_layers = [LSTMLayer(
            hidden_units, rate=rate) for _ in range(n_layers)]
    
    def call(
        self, x_input, c_prev, h_prev, training=True):
        c_next = []
        h_next = []

        lstm_input = x_input
        for m in range(self.n_layers):
            output_tuple = self.lstm_layers[m](
                lstm_input, c_prev[m], h_prev[m], training=training)
            
            c_next.append(
                tf.expand_dims(output_tuple[0], axis=0))
            h_next.append(
                tf.expand_dims(output_tuple[1], axis=0))
            
            res_output = output_tuple[1]
            if self.res_conn:
                res_output += lstm_input
            lstm_input = res_output
        
        lstm_output = output_tuple[1]
        c_next_stacked = tf.concat(c_next, axis=0)
        h_next_stacked = tf.concat(h_next, axis=0)
        return (c_next_stacked, h_next_stacked, lstm_output)

class LSTM(tf.keras.Model):
    def __init__(
        self, n_layers, hidden_units, vocab_size, 
        max_seq_length, res_conn=True, rate=0.1):
        super(LSTM, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_length
        self.hidden_units = hidden_units
        
        # Dropout layer. #
        self.dropout = tf.keras.layers.Dropout(rate)

        # Vocabulary Embedding. #
        self.dec_embed = tf.keras.layers.Embedding(
            vocab_size, hidden_units, name="embedding")

        # LSTM Network. #
        self.lstm_model = LSTMDecoder(
            n_layers, hidden_units, res_conn=res_conn, rate=rate)
    
    def call(self, x, c_prev, h_prev, training=True):
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = self.dropout(
            x_tok_embed, training=training)
        
        output_tuple = self.lstm_model(
            x_tok_embed, c_prev, h_prev, training=training)
        
        c_next = output_tuple[0]
        h_next = output_tuple[1]

        # Get the embedding matrix. #
        x_vocab = tf.range(self.vocab_size)
        w_embed = self.dec_embed(x_vocab)

        dec_logit = tf.matmul(
            output_tuple[2], w_embed, transpose_b=True)
        return (c_next, h_next, dec_logit)
    
    def forward(self, s_prev, x):
        c_prev = s_prev[0]
        h_prev = s_prev[1]

        x_tok_embed  = self.dec_embed(x)
        x_tok_embed = self.dropout(
            x_tok_embed, training=s_prev[3])
        
        lstm_tuple = self.lstm_model(
            x_tok_embed, c_prev, h_prev, training=s_prev[3])
        
        output_tuple = (
            lstm_tuple[0], lstm_tuple[1], lstm_tuple[2], s_prev[3])
        return output_tuple
    
    def decode(
        self, x, c_initial=None, h_initial=None, 
        return_states=True, training=True):
        input_shape = x.shape
        batch_size  = input_shape[0]
        zero_shape  = [
            self.n_layers, batch_size, self.hidden_units]
        
        # Initialise the states. #
        if c_initial is None:
            c_init = tf.zeros(zero_shape, dtype=tf.float32)
        else:
            c_init = c_initial
        
        if h_initial is None:
            h_init = tf.zeros(zero_shape, dtype=tf.float32)
        else:
            h_init = h_initial
        o_init = tf.zeros(zero_shape[1:], dtype=tf.float32)

        # Reshape the input to seq_len by batch. #
        x_input = tf.transpose(x, [1, 0])

        init_states = (
            c_init, h_init, o_init, training)
        lstm_states = tf.scan(
            self.forward, x_input, 
            init_states, parallel_iterations=1)
        
        # Get the embedding matrix. #
        x_vocab = tf.range(self.vocab_size)
        w_embed = self.dec_embed(x_vocab)

        dec_hidden = tf.transpose(
            lstm_states[2], [1, 0, 2])
        dec_logits = tf.matmul(
            dec_hidden, w_embed, transpose_b=True)
        
        if return_states:
            dec_c_state = lstm_states[0][-1]
            return (dec_c_state, dec_logits)
        else:
            return dec_logits
    
    def infer(
        self, x, c_initial=None, gen_len=None, sample=True):
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        if gen_len is None:
            gen_len = self.max_seq_len
        
        batch_size = tf.shape(x)[0]
        zero_shape = [
            self.n_layers, batch_size, self.hidden_units]
        
        # Initialise the states. #
        if c_initial is None:
            c_prev = tf.zeros(zero_shape, dtype=tf.float32)
        else:
            c_prev = c_initial
        h_prev = tf.zeros(zero_shape, dtype=tf.float32)

        for step in range(gen_len):
            curr_inputs = tf.concat(infer_ids, axis=1)
            
            next_tuple = self.call(
                curr_inputs[:, -1], 
                c_prev, h_prev, training=False)
            
            # Update the states. #
            c_prev = next_tuple[0]
            h_prev = next_tuple[1]
            
            tmp_logit = next_tuple[2]
            if sample:
                tmp_probs  = tf.nn.softmax(
                    tmp_logit, axis=1).numpy()[0, :]
                tmp_sample = np.random.choice(
                    self.vocab_size, p=tmp_probs)
                tmp_sample = tf.expand_dims(tf.constant(
                    tmp_sample, dtype=tf.int32), axis=0)
            else:
                tmp_sample = tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32)
            
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], lambda: tmp_sample)
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
