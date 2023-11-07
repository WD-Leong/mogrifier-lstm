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
