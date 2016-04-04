# -*- coding: utf-8 -*-
from gensim.models.word2vec import Word2Vec
import numpy as np
from string import punctuation
import re
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression


def clean_tweet(tweet):
    tknzr = TweetTokenizer()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet.lower())
    tweet = ' '.join(tweet.split())
    words = tknzr.tokenize(tweet)
    words = [''.join(c for c in s if c not in punctuation) for s in words]
    words = [s for s in words if s]
    sent = " ".join(words)
    return sent


def read_file(file_name):
    with open(file_name, 'r') as f:
        content = f.readlines()

    tweets = []
    for line in content:
        try:
            sent = clean_tweet(line)
            if len(sent) > 2:
                tweets.append(sent)
        except UnicodeDecodeError:
            continue
    return tweets


train_pos_tweets = read_file('test_pos.txt')
train_neg_tweets = read_file('train_neg.txt')

test_pos_tweets = read_file('test_pos.txt')
test_neg_tweets = read_file('test_neg.txt')

train_y = np.concatenate((np.ones(len(train_pos_tweets)), np.zeros(len(train_neg_tweets))))
#train_x = np.concatenate(train_pos_tweets, train_neg_tweets)
train_x = np.asarray(train_pos_tweets + train_neg_tweets)
# print train_x
# print train_y

test_y = np.concatenate((np.ones(len(test_pos_tweets)), np.zeros(len(test_neg_tweets))))
#train_x = np.concatenate(train_pos_tweets, train_neg_tweets)
test_x = np.asarray(test_pos_tweets + test_neg_tweets)

sentences = np.concatenate((train_x, test_x))
print sentences

num_features = 300    # Word vector dimensionality
min_word_count = 4   # Minimum word count
num_workers = 2       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
imdb_w2v = word2vec.Word2Vec(sentences, workers=num_workers,
            size=num_features, min_count = min_word_count,
            window = context, sample = downsampling)
#imdb_w2v.init_sims(replace=True)
imdb_w2v.save('word2vec_model')

def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            print "No such word in model: %s" % word
            continue
    if count != 0:
        vec /= count
    return vec


train_vecs = np.concatenate([buildWordVector(z, num_features) for z in train_x])
train_vecs = scale(train_vecs)
test_vecs = np.concatenate([buildWordVector(z, num_features) for z in test_x])
test_vecs = scale(test_vecs)
print train_vecs

classifier = LogisticRegression()
classifier.fit(train_vecs, train_y)
print "Accuracy:", classifier.score(test_vecs, test_y)
