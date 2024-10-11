# Mogrifier-LSTM

This is a Tensorflow Implementation of [Mogrifier LSTM](https://arxiv.org/abs/1909.01792). To train the model on the [Fraser short jokes dataset](https://huggingface.co/datasets/Fraser/short-jokes), first download the dataset and process it by sub-word tokens via
```
python process_fraser_jokes_subword.py
```
followed by
```
python train_fraser_jokes_sw_tf_ver2_lstm.py
```
to train the model. After training the model, run
```
python infer_fraser_jokes_sw_tf_ver2_lstm.py
```
to perform inference.

The model can also be trained on the [movie dialog dataset](https://github.com/Abonia1/TF-Chatbot/tree/master/data). Prior to training the mode, first download the data, then run the script
```
python process_movie_dialog_subword.py
```
to process the corpus into its sub-word tokens. To train the model on the sub-word vocabulary, run
```
python train_movie_dialog_sw_tf_ver2_lstm.py
```
followed by
```
python infer_movie_dialog_sw_tf_ver2_lstm.py
```
to perform inference.

Some examples of the inferred response of the trained model are:
```
Enter input phrase: what time is it?

Input Phrase:
what time is it?
Generated Response:
SOS seven - thirty . EOS
--------------------------------------------------
Enter input phrase: how much does it cost?

Input Phrase:
how much does it cost?
Generated Response:
SOS two hundred dollars . EOS
--------------------------------------------------
Enter input phrase: where are we going?

Input Phrase:
where are we going?
Generated Response:
SOS to the hotel . to register . EOS
--------------------------------------------------
```

## Additional Note

The original Mogrifier LSTM was modified to include elements of the PaLM architecture. A feed-forward layer is added in parallel to the layer-normalized inputs and a residual connection adding the Mogrifier LSTM output, the feed-forward output and the un-normalized input is included as the final output at each layer. 

The modification of the Mogrifier LSTM was also applied to the RNN to try to improve the vanilla RNN performance. Memory footprint is improved by using tf.scan and run-time is improved by using tf.function from the `tf_ver2_mog_rnn_scan.py` and `tf_ver2_mog_lstm.py` files. Import the libraries accordingly to use them.
