
import time
import numpy as np
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_mogrifier_lstm as tf_lstm

# Model Parameters. #
seq_length = 31
num_layers = 3
prob_keep  = 0.9
hidden_size = 256

model_ckpt_dir  = "../TF_Models/dialogue_subword_mog_lstm"
train_loss_file = "train_loss_dialogue_subword_mog_lstm.csv"

# Load the data. #
tmp_pkl_file = "../Data/movie_dialogs/movie_dialogues_sw.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_vocab)
print("Subword Vocabulary Size:", str(vocab_size)+".")

SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the Transformer. #
print("Building the GPT Model.")
start_time = time.time()

lstm_model = tf_lstm.LSTM(
    num_layers, hidden_size, 
    vocab_size, seq_length, rate=0.1)
lstm_optim = tf.keras.optimizers.Adam()

elapsed_time = (time.time()-start_time) / 60
print("LSTM Model Built", "(" + str(elapsed_time), "mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    lstm_model=lstm_model, 
    lstm_optim=lstm_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

# Train the Transformer model. #
tmp_test_in = np.zeros(
    [1, seq_length], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)

print("-" * 50)
print("Testing the LSTM Network", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

# Update the neural network's weights. #
while True:
    tmp_in_phrase = input("Enter input phrase: ")
    tmp_in_phrase = tmp_in_phrase.lower().strip()
    
    if tmp_in_phrase == "":
        break
    else:
        tmp_sw_toks = bpe.bp_encode(
            tmp_in_phrase, subword_vocab, subword_2_idx)
        n_sw_tokens = len(tmp_sw_toks)
        tmp_sw_toks += [SOS_token]
        
        tmp_test_in = np.array(
            tmp_sw_toks, dtype=np.int32).reshape((1, -1))
        
        tmp_infer  = lstm_model.infer(
            tmp_test_in, seq_length).numpy()[0]
        gen_output = tmp_infer[n_sw_tokens:]
        gen_phrase = bpe.bp_decode(
            tmp_infer, idx_2_subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        gen_output = bpe.bp_decode(
            gen_output, idx_2_subword)
        gen_output = " ".join(
            gen_output).replace("<", "").replace(">", "")
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Response:")
        print(gen_output)
        print("-" * 50)
