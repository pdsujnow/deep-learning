#!/bin/bash

python train.py \
--pos_file=../sentiment140/train.pos.txt \
--neg_file=../sentiment140/train.neg.txt \
--word2vec_file=./word2vec/model.ckpt-2265405 \
--vocab_file=./word2vec/vocab.txt \
--filter_sizes="1,2,3,4" \
--num_epoch=50 \
--checkpoint_dir=./checkpoints
