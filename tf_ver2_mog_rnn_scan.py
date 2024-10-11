# Import the libraries. #
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

# Mogrifier RNN Layer. #
class RNNLayer(tf.keras.layers.Layer):
    def __init__(
        self, hidden_units, n_rounds=3, res_conn=True, rate=0.1):
        super(RNNLayer, self).__init__()
        self.rate = rate
        self.n_rounds = n_rounds
        self.res_conn = res_conn
        self.hidden_units = hidden_units

        # RNN weights. #
        self.Wh = tf.keras.layers.Dense(hidden_units)
        self.Wy = tf.keras.layers.Dense(hidden_units)
        self.Wm = [tf.keras.layers.Dense(
            hidden_units) for _ in range(n_rounds)]

        # RNN biases. #
        self.bh = self.add_weight(
            "bh", shape=(hidden_units), initializer="zeros")
        self.by = self.add_weight(
            "by", shape=(hidden_units), initializer="zeros")

        # Pre-Normalization Layer. #
        self.l_norm  = LayerNorm(hidden_units, epsilon=1.0e-6)
        self.dropout = tf.keras.layers.Dropout(rate)

        # Feed forward Layer. #
        self.ffw_1 = tf.keras.layers.Dense(4*hidden_units)
        self.ffw_2 = tf.keras.layers.Dense(hidden_units)
    
    @tf.function
    def call(
        self, x_curr, h_prev, training=True):
        x_norm = self.l_norm(x_curr)

        # Assign the Mogrifier states. #
        x_mog_prev = x_norm
        h_mog_prev = h_prev
        
        # Mogrifier Layer before RNN Layer. #
        for n_rd in range(self.n_rounds):
            if n_rd % 2 == 0:
                h_prev = tf.multiply(
                    h_mog_prev, 2*tf.nn.sigmoid(
                        self.Wm[n_rd](x_mog_prev)))
                h_mog_prev = h_prev
            else:
                x_prev = tf.multiply(
                    x_mog_prev, 2*tf.nn.sigmoid(
                        self.Wm[n_rd](h_mog_prev)))
                x_mog_prev = x_prev

        # Assign the RNN inputs and states. #
        x_norm = x_mog_prev
        h_prev = h_mog_prev
        
        x_conc = tf.concat([x_norm, h_prev], axis=1)
        h_curr = tf.nn.tanh(self.Wh(x_conc) + self.bh)
        y_curr = self.Wy(h_curr) + self.by
        
        # Feed forward. #
        y_ffwd = self.ffw_2(tf.nn.relu(self.ffw_1(x_norm)))
        
        # Residual Connection. #
        res_output = x_curr + y_curr + y_ffwd
        res_output = self.dropout(
            res_output, training=training)
        return (h_curr, res_output)

class RNNNetwork(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, hidden_units, 
        rate=0.1, n_rounds=3, res_conn=True):
        super(RNNNetwork, self).__init__()
        
        self.rate = rate
        self.n_rounds = n_rounds
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.hidden_units = hidden_units
        
        # Decoder Layers. #
        self.dec_layers = [RNNLayer(
            hidden_units, rate=rate, 
            n_rounds=n_rounds) for _ in range(n_layers)]
    
    def call(
        self, x_input, h_prev, training=True):
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
        max_seq_length, n_rounds=3, rate=0.1, res_conn=True):
        super(RNN, self).__init__()
        
        self.rate = rate
        self.n_rounds = n_rounds
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_length
        self.hidden_units = hidden_units

        # Dropout layer. #
        self.dropout = tf.keras.layers.Dropout(rate)

        # Vocabulary Embedding. #
        self.dec_embed = tf.keras.layers.Embedding(
            vocab_size, hidden_units, name="vocab_embedding")

        # RNN Network. #
        self.rnn_model = RNNNetwork(
            n_layers, hidden_units, rate=rate, 
            n_rounds=n_rounds, res_conn=res_conn)
    
    def call(self, x, h_prev, training=True):
        x_tok_embed  = self.dec_embed(x)
        x_tok_embed  = self.dropout(
            x_tok_embed, training=training)
        output_tuple = self.rnn_model(
            x_tok_embed, h_prev, training=training)
        
        # Extract the embedding matrix. #
        x_vocab_idx = tf.range(self.vocab_size)
        W_embedding = self.dec_embed(x_vocab_idx)

        h_current = output_tuple[0]
        dec_logit = tf.matmul(
            output_tuple[1], W_embedding, transpose_b=True)
        return (h_current, dec_logit)

    # For the prefix sum. #
    def forward(self, s_prev, x):
        h_prev = s_prev[0]

        x_tok_embed  = self.dec_embed(x)
        x_tok_embed = self.dropout(
            x_tok_embed, training=True)
        
        rnn_tuple = self.rnn_model(
            x_tok_embed, h_prev, training=True)
        return (rnn_tuple[0], rnn_tuple[1])
    
    # Use the prefix sum to compute during training. #
    def decode(self, x, training=True):
        input_shape = x.shape
        batch_size  = input_shape[0]
        zero_shape  = [
            self.n_layers, batch_size, self.hidden_units]
        
        # Initialise the states. #
        h_init = tf.zeros(zero_shape, dtype=tf.float32)
        o_init = tf.zeros(zero_shape[1:], dtype=tf.float32)

        # Reshape the input to seq_len by batch. #
        x_input = tf.transpose(x, [1, 0])

        # Extract the embedding matrix. #
        x_vocab_idx = tf.range(self.vocab_size)
        W_embedding = self.dec_embed(x_vocab_idx)
        
        # Run the prefix sum algorithm. #
        init_state = (h_init, o_init)
        rnn_states = tf.scan(
            self.forward, x_input, 
            init_state, parallel_iterations=1)
        dec_states = tf.transpose(rnn_states[1], [1, 0, 2])
        dec_logits = tf.matmul(
            dec_states, W_embedding, transpose_b=True)
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
