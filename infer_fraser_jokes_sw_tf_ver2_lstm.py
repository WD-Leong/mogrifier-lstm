
import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_mogrifier_lstm as tf_lstm

# Model Parameters. #
seq_length = 51
num_layers = 3
num_rounds = 5
seq_length = 51

prob_keep = 0.9
hidden_size = 256

model_ckpt_dir  = "../TF_Models/keras_lstm_sw_fraser_jokes"
train_loss_file = "train_loss_keras_lstm_sw_fraser_jokes.csv"

# Load the data. #
tmp_pkl_file = "../Data/jokes/short_jokes.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)

    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_2_idx)
print("Vocabulary Size:", str(vocab_size)+".")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

num_data  = len(full_data)
SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]
print("Total of", str(len(full_data)), "rows loaded.")

# Build the GPT. #
print("Building the GPT Keras Model.")
start_time = time.time()

lstm_model = tf_lstm.MogrifierLSTM(
    num_layers, hidden_size, vocab_size, 
    seq_length, n_rounds=num_rounds, rate=1.0-prob_keep)
lstm_optim = tfa.optimizers.AdamW(weight_decay=1.0e-4)

elapsed_time = (time.time() - start_time) / 60
print("GPT Keras Model Built", 
      "(" + str(elapsed_time) + " mins).")

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

train_loss_df = pd.read_csv(train_loss_file)
train_loss_list = [tuple(
    train_loss_df.iloc[x].values) \
    for x in range(len(train_loss_df))]

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)

print("-" * 50)
print("Inferring the LSTM Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
while True:
    tmp_phrase = input("Enter prompt: ")
    tmp_phrase = tmp_phrase.lower().strip()

    if tmp_phrase == "":
        break
    else:
        i_encode = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword_2_idx)
        n_input  = len(i_encode)
        i_decode = bpe.bp_decode(i_encode, idx_2_subword)
        
        tmp_test_in = np.array(
            i_encode, dtype=np.int32)
        tmp_test_in = tmp_test_in.reshape((1, -1))

        gen_ids = tf_lstm.infer(
            lstm_model, tmp_test_in, 
            gen_len=seq_length, sample=False)
        gen_phrase = bpe.bp_decode(
            gen_ids.numpy()[0], idx_2_subword)
        
        print("Input Phrase:")
        print(" ".join(i_decode).replace("<", "").replace(">", ""))
        print("Generated Phrase:")
        print(" ".join(gen_phrase).replace("<", "").replace(">", ""))
        print("-" * 50)

