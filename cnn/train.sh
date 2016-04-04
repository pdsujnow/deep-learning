#!/bin/bash

python train.py \
--pos_file=../sentiment140/train.pos.txt \
--neg_file=../sentiment140/train.neg.txt \
--model_file=./word2vec/model.ckpt-2265405 \
--vocab_file=./word2vec/vocab.txt \
--out_dir=/home/zxing01/cnn-output/