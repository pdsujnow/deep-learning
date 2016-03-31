#!/bin/bash

THEANO_FLAGS="floatX=float32" OMP_NUM_THREADS=12 python tweet_lstm.py
