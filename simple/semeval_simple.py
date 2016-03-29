

import nltk
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.probability import FreqDist
import random
import csv
from nltk.tokenize import TweetTokenizer
from string import punctuation
import re
import sys
import os.path
from six.moves import cPickle


def clean_tweet(tweet):
    tknzr = TweetTokenizer()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet.lower())
    tweet = ' '.join(tweet.split())
    words = tknzr.tokenize(tweet)
    words = [''.join(c for c in s if c not in punctuation) for s in words]
    words = [s for s in words if s]
    sent = " ".join(words)
    return sent


def load_file(file_name):
    tweets = []
    with open(file_name, 'r') as f:
        contents = f.readlines()
    for line in contents:
        index = line.find('\t')
        if index == -1:
            raise RuntimeError('no tab sign')
        if line.find('\t', index + 1) != -1:
            raise RuntimeError('multiple tab sign')

        row = line.split('\t')
        if row[0] == "neutral":
            continue
        sentence = clean_tweet(row[1])
        tweets.append((sentence, row[0]))

    return tweets


def build_tweet_model(file_name):
    data = load_file(file_name)
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
    # word_features = wordlist.keys() # careful here
    word_features = [w for (w, c) in wordlist.most_common(2000)] #use most_common() if you want to select the most frequent words
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

    training_set = []
    testing_set = []

    if os.path.isfile('semeval.pkl'):
        print "Load pkl file..."
        f = open('semeval.pkl', 'r')
        training_set = cPickle.load(f)
        testing_set = cPickle.load(f)
    else:
        print "Build training and testing features..."
        train_tweets = build_tweet_model("../semeval/train.tsv")
        test_tweets = build_tweet_model("../semeval/test.tsv")

        #all_words = get_words_in_tweets(train_tweets)
        word_features = get_word_features(get_words_in_tweets(train_tweets))

        training_set = [(extract_features(d, word_features), c) for (d,c) in train_tweets]
        testing_set = [(extract_features(d, word_features), c) for (d,c) in test_tweets]

        print "Write training and testing sets to pkl file..."
        f = open('semeval.pkl', 'w')
        cPickle.dump(training_set, f)
        cPickle.dump(testing_set, f)
        f.close()
    
    print "Begin classification..."
    classifier = None
    if (sys.argv[1] == '-nb'):
        classifier = nltk.NaiveBayesClassifier.train(training_set)  # 0.721470019342
    elif (sys.argv[1] == '-dt'):
        classifier = nltk.DecisionTreeClassifier.train(training_set)
    elif (sys.argv[1] == '-me'):
        classifier = nltk.MaxentClassifier.train(training_set, algorithm='iis', max_iter=3)
    elif (sys.argv[1] == '-svm'):
        classif = SklearnClassifier(LinearSVC())  # 0.661508704062
        classifier = classif.train(training_set)  # 0.727272727273
    else:
        print "Unknown classifier"

    print nltk.classify.accuracy(classifier, testing_set)
