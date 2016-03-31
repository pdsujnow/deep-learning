# -*- coding: utf-8 -*-
"""
    sentiment140_simple.py
    ~~~~~~~~~~~~~~~~~~~~

    This file defines the methods to implement basic sentiment analysis on Sentiment 140 tweets data.
    The classifiers used in this file include:
        Naive Bayes
        Decision Tree
        Maximum Entropy
        Support Vector Machine
    The script is based on the Python script provided by Dr. Edmund Yu, which is originally used to
    implement sentiment analysis on movie reviews.

    Usage of this script:
        python semeval_simple.py -[nb/dt/me/svm]
"""


import nltk
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.probability import FreqDist
import random
from nltk.tokenize import TweetTokenizer
from string import punctuation
import re
import sys
import os.path
from six.moves import cPickle


def clean_tweet(tweet):
    tknzr = TweetTokenizer()
    try:
        tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet.lower())
        tweet = ' '.join(tweet.split())
        words = tknzr.tokenize(tweet)
        words = [''.join(c for c in s if c not in punctuation) for s in words]
        words = [s for s in words if s]
    except UnicodeDecodeError:
        return []

    sent = " ".join(words)
    return sent


def load_file(file_name, polarity, sample_size=None):
    tweets = []
    with open(file_name, 'r') as f:
        contents = f.readlines()

    if sample_size > 0:
        random.shuffle(contents)
        contents = contents[0:sample_size]

    for line in contents:
        sentence = clean_tweet(line)
        if len(sentence) > 0:
            tweets.append((sentence, polarity))

    return tweets


def build_tweet_model(file_name, polarity, sample_size=None):
    data = load_file(file_name, polarity, sample_size)
    tweets = []
    for (words, sentiment) in data:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append((words_filtered, sentiment))
    return tweets


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words += words
    return all_words


def get_word_features(wordlist):
    wordlist = FreqDist(wordlist)
    # use most_common() if you want to select the most frequent words
    word_features = [w for (w, c) in wordlist.most_common(2000)]
    return word_features


def extract_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "-nb  Naive Bayes Classifier"
        print "-dt  Decision Tree Classifier"
        print "-me  Max Entropy Classifier"
        print "-svm Support Vector Machine"
        sys.exit(1)

    print "Build training and testing features..."
    train_pos_tweets = build_tweet_model("../sentiment140/train.pos.txt", "P", 5000)
    train_neg_tweets = build_tweet_model("../sentiment140/train.neg.txt", "N", 5000)
    test_pos_tweets = build_tweet_model("../sentiment140/test.pos.txt", "P")
    test_neg_tweets = build_tweet_model("../sentiment140/test.neg.txt", "N")
    train_tweets = train_pos_tweets + train_neg_tweets
    test_tweets = test_pos_tweets + test_neg_tweets

    #all_words = get_words_in_tweets(train_tweets)
    word_features = get_word_features(get_words_in_tweets(train_tweets))

    training_set = [(extract_features(d, word_features), c) for (d,c) in train_tweets]
    testing_set = [(extract_features(d, word_features), c) for (d,c) in test_tweets]

    print "Begin classification..."
    classifier = None
    if (sys.argv[1] == '-nb'):
        print "Naive Bayes is used..."
        classifier = nltk.NaiveBayesClassifier.train(training_set)
    elif (sys.argv[1] == '-dt'):
        print "Decision Tree is used..."
        classifier = nltk.DecisionTreeClassifier.train(training_set)
    elif (sys.argv[1] == '-me'):
        print "Max Entropy is used..."
        classifier = nltk.MaxentClassifier.train(training_set, algorithm='iis', max_iter=3)
    elif (sys.argv[1] == '-svm'):
        print "Support Vector Machine is used..."
        classif = SklearnClassifier(LinearSVC())
        classifier = classif.train(training_set)
    else:
        print "Unknown classifier"

    print "Accuracy:", nltk.classify.accuracy(classifier, testing_set)
