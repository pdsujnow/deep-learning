#!/bin/bash

THEANO_FLAGS="floatX=float32" OMP_NUM_THREADS=12 python -u tweet_lstm.py > lstm.log
