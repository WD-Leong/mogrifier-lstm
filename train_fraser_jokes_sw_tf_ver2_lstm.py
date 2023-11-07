
import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_mogrifier_lstm as tf_lstm

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_encode, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = x_encode[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            output_logits = tf_lstm.decode(
                model, tmp_encode, training=True)
            
            tmp_losses = tf.reduce_sum(tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=output_logits), axis=1))
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return avg_losses

# Model Parameters. #
prob_keep  = 0.9
batch_size = 128
sub_batch  = 128
num_layers = 3
num_rounds = 5
seq_length = 51

gradient_clip = 1.00
maximum_iter  = 2000
restore_flag  = True
save_step     = 250
warmup_steps  = 50000
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75

hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 250

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
print("Vocabulary Size:", str(vocab_size) + ".")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Extract the data and its assets. #
tmp_data = []
for tmp_row in full_data:
    if len(tmp_row) > 1 and \
        len(tmp_row) <= seq_length:
        tmp_data.append(tmp_row)

num_data  = len(tmp_data)
SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Build the LSTM network. #
print("Building the LSTM Keras Model.")
start_time = time.time()

lstm_model = tf_lstm.MogrifierLSTM(
    num_layers, hidden_size, vocab_size, 
    seq_length, n_rounds=num_rounds, rate=1.0-prob_keep)
lstm_optim = tfa.optimizers.AdamW(weight_decay=1.0e-4)

# Initialize the model. #
init_zeros_in = tf.zeros(
    [batch_size], dtype=tf.int32)
init_c_states = tf.zeros([
    num_layers, batch_size, hidden_size], dtype=tf.float32)
init_h_states = tf.zeros([
    num_layers, batch_size, hidden_size], dtype=tf.float32)
init_outputs = lstm_model(
    init_zeros_in, init_c_states, init_h_states)
del init_zeros_in, init_c_states, init_h_states, init_outputs
print(lstm_model.summary())

elapsed_time = (time.time()-start_time) / 60
print("LSTM Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    lstm_model=lstm_model, 
    lstm_optim=lstm_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
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
else:
    print("Training a new model.")
    train_loss_list = []

# Train the LSTM model. #
tmp_out_seq = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    learning_rate = max(
        anneal_rate**(n_iter // anneal_step)*initial_lr, 1.0e-5)

print("-" * 50)
print("Training the LSTM Network", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            anneal_factor = np.power(
                anneal_rate, int(n_iter / anneal_step))
            learning_rate = \
                max(anneal_factor*initial_lr, 1.0e-6)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_out_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_p_idx = tmp_data[tmp_index] + [EOS_token]
        
        n_input = len(tmp_p_idx)
        tmp_out_seq[n_index, :n_input] = tmp_p_idx
        del tmp_p_idx
    
    # Set the training data. #
    tmp_input  = tmp_out_seq[:, :-1]
    tmp_output = tmp_out_seq[:, 1:]
    
    tmp_input  = tmp_out_seq[:, :-1]
    tmp_output = tmp_out_seq[:, 1:]
    
    tmp_loss = sub_batch_train_step(
        lstm_model, sub_batch, tmp_input, tmp_output, 
        lstm_optim, learning_rate=learning_rate)
    
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60.0
        
        sample_test = np.random.choice(num_data)
        tmp_p_index = tmp_data[sample_test]
        
        in_phrase = bpe.bp_decode(
            tmp_p_index, idx_2_subword)
        in_phrase = " ".join(in_phrase).replace(
            "<", "").replace(">", "")
        
        n_tokens = len(tmp_p_index)
        n_sample = min(np.random.randint(
            1, high=n_tokens-1), int(n_tokens/2))
        
        tmp_test_in = np.array(tmp_p_index[:n_sample])
        tmp_test_in = np.expand_dims(tmp_test_in, axis=0)
        
        gen_tokens = tf_lstm.infer(
            lstm_model, tmp_test_in, 
            gen_len=seq_length, sample=False)
        gen_phrase = bpe.bp_decode(
            gen_tokens[0].numpy(), idx_2_subword)
        gen_phrase = " ".join(gen_phrase).replace(
            "<", "").replace(">", "")
        
        test_phrase = bpe.bp_decode(
            tmp_p_index[:n_sample], idx_2_subword)
        test_phrase = " ".join(test_phrase).replace(
            "<", "").replace(">", "")
        del tmp_p_index

        print("Iteration", str(n_iter)+".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip)+".")
        print("Learning Rate:", str(learning_rate)+".")
        print("Average Loss:", str(avg_loss)+".")
        
        print("")
        print("Input Phrase:")
        print(test_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("Actual Phrase:")
        print(in_phrase)
        del n_sample, tmp_test_in
        
        train_loss_list.append((n_iter, avg_loss))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)
