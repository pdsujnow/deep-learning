
# CNN for Twitter Sentiment

This package works on Linux and Mac OS X. 

## Files

`readme.txt` -- this readme file.

`data_helpers.py` -- some helper functions for data processsing.

`text_cnn.py` -- TextCNN class that defines the network structure.

`train.py` -- python script of the training procedure.

`eval.py` -- python script of the test procedure.

`train.sh` -- example bash script for running train.py, please modify the parameter values before use.

`eval.sh` -- example bash script for running eval.py, please modify the parameter values before use.

`word2vec/word2vec_optimized.py` -- code for training a word2vec model, taken from TensorFlow source code.

`word2vec/vocab.txt` -- vocabulary file contains all the words recognizable by the word2vec model

`word2vec/model.zip` & `word2vec/model.z01` -- compressed files of a pretrained word2vec model

`word2vec/text8.zip` -- compressed file of the training data for word2vec model

`word2vec/questions-words.txt` -- test data for word2vec model

## Setup

1. Install TensorFlow following the [official tutorial](https://www.tensorflow.org/versions/master/get_started/os_setup.html)

2. Install NLTK using easy_install or pip.

3. If use the pretrained word2vec model, decompress the zip file using the following command:

```
zip -FF model.zip --out model-full.zip; unzip model-full.zip
```
4. The above command produces the model.ckpt-2265405 file, a pretrained word2vec model. 2265405 is the number of steps used in training the model. With this pretrained word2vec, please go directly to section 4 -- Train CNN Model.


## Train word2vec Model

1. The word embedding model word2vec is needed by the CNN model, so it needs to be pretrained if necessary.

2. To train a word2vec model, use the following commands:

```
cd word2vec; unzip text8.zip
python word2vec_optimized.py \
--train_data=text8 \
--eval_data=questions-words.txt \
--save_path=.
```
3. After a long time of training, a model file named "model.ckpt-XXXXXXX" (XXXXXXX is a number)and vocabulary file vocab.txt, among others, will be generated. These two files are needed by the CNN model. 

## Train CNN Model

1. The following descriptions assumes the pretrained word2vec model is used. If it is retrained, replace "model.ckpt-2265405" with the new file name.

2. The python script train.py is the training procedure. To get help info on the parameters, use the following command:

```
python train.py --help
```
3. To train a model with the default parameters, modify and run the train.sh bash script or use the following command:

```
python train.py \
--pos_file=../sentiment140/train.pos.txt \
--neg_file=../sentiment140/train.neg.txt \
--word2vec_file=./word2vec/model.ckpt-2265405 \
--vocab_file=./word2vec/vocab.txt \
--checkpoint_dir=./checkpoints
```

## Test CNN Model

1. The python script eval.py is the test procedure. To get help info on the parameters, use the following command:

```
python eval.py --help
```
2. To test a saved model with the default parameters, modify and run the eval.sh bash script or use the following command:

```
python eval.py \
--pos_file=../sentiment140/test.pos.txt \
--neg_file=../sentiment140/test.neg.txt \
--vocab_file=./word2vec/vocab.txt \
--checkpoint_dir=./checkpoints
```
