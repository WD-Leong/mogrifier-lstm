
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, LayerNormalization, Dense)

# LSTM Layer. #
class MogrifierLSTMLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, n_rounds=5, rate=0.1):
        super(MogrifierLSTMLayer, self).__init__()
        self.rate = rate
        self.n_rounds = n_rounds
        self.hidden_units = hidden_units

        # LSTM weights. #
        self.Wf = tf.keras.layers.Dense(hidden_units)
        self.Wi = tf.keras.layers.Dense(hidden_units)
        self.Wo = tf.keras.layers.Dense(hidden_units)
        self.Wc = tf.keras.layers.Dense(hidden_units)

        self.Uf = tf.keras.layers.Dense(hidden_units)
        self.Ui = tf.keras.layers.Dense(hidden_units)
        self.Uo = tf.keras.layers.Dense(hidden_units)
        self.Uc = tf.keras.layers.Dense(hidden_units)
        
        # Mogrifier weights. #
        self.W_mog = [Dense(
            hidden_units) for _ in range(n_rounds)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
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

        # LSTM Layer after Mogrifier Layer. #
        input_gate  = tf.nn.sigmoid(tf.add(
            self.Wi(x_curr), self.Ui(h_prev)))
        forget_gate = tf.nn.sigmoid(tf.add(
            self.Wf(x_curr), self.Uf(h_prev)))
        output_gate = tf.nn.sigmoid(tf.add(
            self.Wo(x_curr), self.Uo(h_prev)))
        
        c_tilde = tf.nn.tanh(tf.add(
            self.Wc(x_curr), self.Uc(h_prev)))
        
        c_next = tf.add(
            tf.multiply(forget_gate, c_prev), 
            tf.multiply(input_gate, c_tilde))
        h_next = tf.multiply(output_gate, tf.nn.tanh(c_next))
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
            LayerNormalization() for _ in range(n_layers)]
        
        # Decoder Layers. #
        self.dec_layers = [
            MogrifierLSTMLayer(
                hidden_units, rate=rate, 
                n_rounds=self.n_rounds) for _ in range(n_layers)]
    
    def call(self, x_input, c_prev, h_prev, training=True):
        c_next = []
        h_next = []

        layer_output = x_input
        for m in range(self.n_layers):
            layer_input  = self.norm_layers[m](layer_output)
            output_tuple = self.dec_layers[m](
                layer_input, c_prev[m], h_prev[m], training=training)
            
            c_next.append(
                tf.expand_dims(output_tuple[0], axis=0))
            h_next.append(
                tf.expand_dims(output_tuple[1], axis=0))
            
            if self.res_conn:
                layer_output = tf.add(
                    output_tuple[1], layer_input)
            else:
                layer_output = output_tuple[1]
        
        lstm_output = layer_output
        c_next_stacked = tf.concat(c_next, axis=0)
        h_next_stacked = tf.concat(h_next, axis=0)
        return (c_next_stacked, h_next_stacked, lstm_output)

class MogrifierLSTM(tf.keras.Model):
    def __init__(
        self, n_layers, hidden_units, 
        vocab_size, max_seq_length, 
        n_rounds=5, rate=0.1, res_conn=True):
        super(MogrifierLSTM, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.n_rounds = n_rounds
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_length
        self.hidden_units = hidden_units

        # Vocabulary Embedding. #
        self.dec_embed = Embedding(vocab_size, hidden_units)
        
        # Mogrifier LSTM Network. #
        self.lstm_model = MogrifierLSTMNetwork(
            n_layers, hidden_units, rate=rate, 
            n_rounds=n_rounds, res_conn=res_conn)
    
    def call(self, x, c_prev, h_prev, training=True):
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = tf.multiply(
            x_tok_embed, tf.math.rsqrt(
                float(self.hidden_units)))
        
        x_vocab_idx = tf.range(self.vocab_size)
        W_embedding = self.dec_embed(x_vocab_idx)
        output_tuple = self.lstm_model(
            x_tok_embed, c_prev, h_prev, training=training)
        
        c_next = output_tuple[0]
        h_next = output_tuple[1]
        dec_logit = tf.matmul(
            output_tuple[2], W_embedding, transpose_b=True)
        return c_next, h_next, dec_logit

# Place the decoding outside of the model class to optimize #
# the LSTM for loop for faster training.                    #
@tf.function
def decode(tf_model, x, training=True):
    input_shape = x.shape
    batch_size  = input_shape[0]
    dec_length  = input_shape[1]
    zero_shape  = [
        tf_model.n_layers, 
        batch_size, tf_model.hidden_units]
    
    # Initialise the states. #
    c_prev = tf.zeros(zero_shape, dtype=tf.float32)
    h_prev = tf.zeros(zero_shape, dtype=tf.float32)
    
    dec_logits = []
    for t_index in range(dec_length):
        next_tuple = tf_model(
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

# Place the inference outside of the keras model class to #
# optimize the LSTM for loop for faster training.         #
#@tf.function
def infer(
    tf_model, x, gen_len=None, sample=True):
    input_len = x.shape[1]
    infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
    
    if gen_len is None:
        gen_len = tf_model.max_seq_len
    
    batch_size = tf.shape(x)[0]
    zero_shape = [
        tf_model.n_layers, 
        batch_size, tf_model.hidden_units]
    
    # Initialise the states. #
    c_prev = tf.zeros(zero_shape, dtype=tf.float32)
    h_prev = tf.zeros(zero_shape, dtype=tf.float32)

    for step in range(gen_len):
        curr_inputs = tf.concat(infer_ids, axis=1)
        
        next_tuple = tf_model(
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
