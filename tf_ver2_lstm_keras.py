
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, LayerNormalization)

# LSTM Layer. #
class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, rate=0.1):
        super(LSTMLayer, self).__init__()
        self.rate = rate
        self.hidden_units = hidden_units

        self.linear = tf.keras.layers.Dense(4*hidden_units)
        self.lnorm  = LayerNormalization(epsilon=1.0e-6)
        self.o_proj = tf.keras.layers.Dense(hidden_units)
        self.ffwd_1  = tf.keras.layers.Dense(4*hidden_units)
        self.ffwd_2  = tf.keras.layers.Dense(hidden_units)
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x_curr, c_prev, h_prev, training=True):
        x_norm = self.lnorm(x_curr)

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
        h_next = tf.multiply(output_gate, tf.nn.tanh(c_next))

        # Feed forward network. #
        x_ffwd = self.ffwd_2(
            tf.nn.relu(self.ffwd_1(x_norm)))
        
        # Residual connection. #
        y_next = x_curr + x_ffwd + self.o_proj(h_next)
        
        c_next = self.dropout(c_next, training=training)
        h_next = self.dropout(h_next, training=training)
        y_next = self.dropout(y_next, training=training)
        return (c_next, h_next, y_next)

class LSTMNetwork(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, 
        hidden_units, res_conn=True, rate=0.1):
        super(LSTMNetwork, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.hidden_units = hidden_units
        
        # Decoder Layers. #
        self.dec_layers = [LSTMLayer(
            hidden_units, rate=rate) for _ in range(n_layers)]
    
    def call(
        self, x_input, c_prev, h_prev, training=True):
        c_next = []
        h_next = []

        prev_input = 0.0
        lstm_input = x_input
        for m in range(self.n_layers):
            c_state = c_prev[m]
            h_state = h_prev[m]
            output_tuple = self.dec_layers[m](
                lstm_input, c_state, h_state, training=training)
            
            c_next.append(
                tf.expand_dims(output_tuple[0], axis=0))
            h_next.append(
                tf.expand_dims(output_tuple[1], axis=0))
            
            res_output = output_tuple[2]
            if self.res_conn:
                res_output += prev_input
            
            prev_input = lstm_input
            lstm_input = res_output
        
        lstm_output = lstm_input
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
        self.dec_embed = Embedding(vocab_size, hidden_units)

        # LSTM Network. #
        self.lstm_model = LSTMNetwork(
            n_layers, hidden_units, 
            res_conn=res_conn, rate=rate)
    
    def call(self, x, c_prev, h_prev, training=True):
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = self.dropout(
            x_tok_embed, training=training)
        
        output_tuple = self.lstm_model(
            x_tok_embed, c_prev, h_prev, training=training)
        
        c_next = output_tuple[0]
        h_next = output_tuple[1]

        x_vocab_idx = tf.range(self.vocab_size)
        W_embedding = self.dec_embed(x_vocab_idx)
        dec_logit = tf.matmul(
            output_tuple[2], W_embedding, transpose_b=True)
        return (c_next, h_next, dec_logit)
    
    @tf.function
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
    
    def infer(
        self, x, gen_len=None, sample=True):
        input_len = tf.shape(x)[1]
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
