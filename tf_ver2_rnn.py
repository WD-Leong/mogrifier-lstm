# Import the libraries. #
import tensorflow as tf

# RNN Layer. #
class RNNLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, res_conn=True, rate=0.1):
        super(RNNLayer, self).__init__()
        self.rate = rate
        self.res_conn = res_conn
        self.hidden_units = hidden_units

        # RNN weights. #
        self.Wh = tf.keras.layers.Dense(hidden_units)
        self.Wx = tf.keras.layers.Dense(hidden_units)
        self.Wy = tf.keras.layers.Dense(hidden_units)

        # RNN biases. #
        self.bh = self.add_weight(
            "bh", shape=(hidden_units), initializer="zeros")
        self.by = self.add_weight(
            "by", shape=(hidden_units), initializer="zeros")

        # Pre-Normalization Layer. #
        self.lnorm = tf.keras.layers.LayerNormalization(epsilon=1.0e-6)

        # Feed forward layers. #
        self.ffw_1 = tf.keras.layers.Dense(4*hidden_units)
        self.ffw_2 = tf.keras.layers.Dense(hidden_units)
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(
        self, x_curr, h_prev, training=True):
        x_norm = self.lnorm(x_curr)
        h_curr = tf.nn.relu(
            self.Wh(h_prev) + self.Wx(x_norm) + self.bh)
        y_curr = self.Wy(h_curr) + self.by
        
        # Feed forward layer. #
        y_ffwd = self.ffw_2(
            tf.nn.relu(self.ffw_1(x_norm)))
        
        # Residual Connection. #
        res_output = tf.add(
            x_curr, y_curr + y_ffwd)
        res_output = self.dropout(
            res_output, training=training)
        return (h_curr, res_output)

class RNNNetwork(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, hidden_units, rate=0.1, res_conn=True):
        super(RNNNetwork, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.hidden_units = hidden_units
        
        # Decoder Layers. #
        self.dec_layers = [RNNLayer(
            hidden_units, rate=rate) for _ in range(n_layers)]
    
    def call(self, x_input, h_prev, training=True):
        h_curr = []
        
        prev_input  = 0.0
        layer_input = x_input
        for m in range(self.n_layers):
            # RNN Layer. #
            output_tuple = self.dec_layers[m](
                layer_input, h_prev[m], training=training)

            h_curr.append(
                tf.expand_dims(output_tuple[0], axis=0))
            
            layer_output = output_tuple[1]
            if self.res_conn:
                layer_output += prev_input
            prev_input  = layer_input
            layer_input = layer_output
        
        rnn_output = layer_output
        h_curr_stacked = tf.concat(h_curr, axis=0)
        return (h_curr_stacked, rnn_output)

class RNN(tf.keras.Model):
    def __init__(
        self, n_layers, hidden_units, vocab_size, 
        max_seq_length, rate=0.1, res_conn=True):
        super(RNN, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_length
        self.hidden_units = hidden_units

        # Vocabulary Embedding. #
        self.dec_embed = tf.keras.layers.Embedding(
            vocab_size, hidden_units, name="vocab_embedding")
        
        # Output projection. #
        self.out_proj = tf.keras.layers.Dense(
            vocab_size, name="output_projection_layer")

        # RNN Network. #
        self.rnn_model = RNNNetwork(
            n_layers, hidden_units, 
            rate=rate, res_conn=res_conn)
    
    def call(self, x, h_prev, training=True):
        x_tok_embed  = self.dec_embed(x)
        output_tuple = self.rnn_model(
            x_tok_embed, h_prev, training=training)
        
        h_curr = output_tuple[0]
        dec_logit = self.out_proj(output_tuple[1])
        return (h_curr, dec_logit)
    
    @tf.function
    def decode(self, x, training=True):
        input_shape = x.shape
        batch_size  = input_shape[0]
        dec_length  = input_shape[1]
        zero_shape  = [
            self.n_layers, batch_size, self.hidden_units]
        
        # Initialise the states. #
        h_prev = tf.zeros(zero_shape, dtype=tf.float32)

        dec_logits = []
        for t_index in range(dec_length):
            next_tuple = self.call(
                x[:, t_index], h_prev, training=training)
            
            # Update the states. #
            h_prev = next_tuple[0]
            
            # Append the output logits. #
            dec_logits.append(
                tf.expand_dims(next_tuple[1], axis=1))
        
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
        h_prev = tf.zeros(zero_shape, dtype=tf.float32)

        for step in range(gen_len):
            curr_inputs = tf.concat(infer_ids, axis=1)
            next_tuple  = self.call(
                curr_inputs[:, -1], h_prev, training=False)
            
            tmp_logit = next_tuple[1]
            if sample:
                tmp_sample = tf.random.categorical(
                    tmp_logit, 1, dtype=tf.int32)[:, 0]
            else:
                tmp_sample = tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32)
            
            # Update the states. #
            h_prev = next_tuple[0]
            
            if step < (input_len-1):
                tmp_index = x[:, step+1]
            else:
                tmp_index = tmp_sample
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
