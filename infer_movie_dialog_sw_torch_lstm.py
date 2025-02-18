# Import the libraries. #
import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import torch
import torch_mog_lstm_ffwd as lstm_module

# Model Parameters. #
seq_length = 31
num_layers = 2

prob_keep = 0.9
prob_drop = 1.0 - prob_keep
hidden_size = 256

model_ckpt_dir  = "../PyTorch_Models/dialog_sw_torch_mog_lstm"
train_loss_file = "train_loss_dialog_sw_torch_mog_lstm.csv"

# Load the data. #
tmp_pkl_file = "../Data/movie_dialogs/"
tmp_pkl_file += "movie_dialogues_sw.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_vocab)
print("Subword Vocabulary Size:", str(vocab_size) + ".")

SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Set the number of threads to use. #
torch.set_num_threads(1)

# Build the GPT network. #
print("Building the LSTM Model.")
start_time = time.time()

lstm_model = lstm_module.LSTM(
    num_layers, hidden_size, vocab_size, 
    seq_length, rate=prob_drop, res_conn=True)
if torch.cuda.is_available():
    lstm_model.to("cuda")

lstm_optim = torch.optim.AdamW(
    lstm_model.parameters(), weight_decay=1.0e-4)

elapsed_time = (time.time()-start_time) / 60
print("LSTM Model Built (" + str(elapsed_time), "mins).")

ckpt = torch.load(model_ckpt_dir)
n_iter = ckpt["step"]

lstm_model.load_state_dict(ckpt['model_state_dict'])
lstm_optim.load_state_dict(ckpt['optim_state_dict'])

train_loss_df = pd.read_csv(train_loss_file)
train_loss_list = [tuple(
    train_loss_df.iloc[x].values) \
    for x in range(len(train_loss_df))]

# GPT model inference. #
print("-" * 50)
print("LSTM Model Inference", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

# Set the model to eval. #
lstm_model.eval()

# Start inferring. #
while True:
    tmp_phrase = input("Enter input: ")
    tmp_phrase = tmp_phrase.lower().strip()

    if tmp_phrase == "":
        break
    else:
        tmp_i_idx = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword_2_idx)
        n_tokens  = len(tmp_i_idx)
        
        tmp_test_in = tmp_i_idx + [SOS_token]
        tmp_test_in = np.array(tmp_test_in).reshape((1, -1))
        
        infer_in = torch.tensor(
            tmp_test_in, dtype=torch.long)
        if torch.cuda.is_available():
            infer_in = infer_in.to("cuda")
        
        tmp_infer = lstm_model.infer(infer_in, k=1)
        if torch.cuda.is_available():
            tmp_infer = tmp_infer.detach().cpu()
        
        gen_tokens = tmp_infer[0].numpy()
        gen_phrase = bpe.bp_decode(
            gen_tokens, idx_2_subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        gen_reply = gen_tokens[n_tokens:]
        gen_reply = bpe.bp_decode(
            gen_reply, idx_2_subword)
        gen_reply = " ".join(
            gen_reply).replace("<", "").replace(">", "")
        del n_tokens

        print("")
        print("Input Phrase:")
        print(tmp_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("Generated Reply:")
        print(gen_reply)
        print("-" * 50)
