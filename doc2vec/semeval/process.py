# -*- coding: utf-8 -*-
"""
    process.py
    ~~~~~~~~~~~~~

    This file defines methods to implement sentiment analysis on Doc2Vec model which is
    trained through build_model.py file.
    This file contains the following classifiers:
        Logistic Regression

    The content of this file is based on the reference:
    https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb
"""


from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from sklearn.linear_model import LogisticRegression


model = Doc2Vec.load('./semeval.d2v')

# train_pos_count = 898
# train_neg_count = 887
# test_pos_count = 100
# test_neg_count = 98

train_pos_count = 2290
# train_neu_count = 3046
train_neg_count = 848
test_pos_count = 342
# test_neu_count = 422
test_neg_count = 175


print "Build training data set..."
train_arrays = numpy.zeros((train_pos_count + train_neg_count, 20))
train_labels = numpy.zeros(train_pos_count + train_neg_count)

for i in range(train_pos_count):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in range(train_neg_count):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[train_pos_count + i] = model.docvecs[prefix_train_neg]
    train_labels[train_pos_count + i] = 0

# prev_count = train_pos_count + train_neg_count
# for i in range(train_neu_count):
#     prefix_train_neu = 'TRAIN_NEU_' + str(i)
#     train_arrays[prev_count + i] = model.docvecs[prefix_train_neu]
#     train_labels[prev_count + i] = 2


print "Build testing data set..."
test_arrays = numpy.zeros((test_pos_count + test_neg_count, 20))
test_labels = numpy.zeros(test_pos_count + test_neg_count)

for i in range(test_pos_count):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_labels[i] = 1

for i in range(test_neg_count):
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[test_pos_count + i] = model.docvecs[prefix_test_neg]
    test_labels[test_pos_count + i] = 0

# prev_count = test_pos_count + test_neg_count
# for i in range(test_neu_count):
#     #print i
#     prefix_test_neu = 'TEST_NEU_' + str(i)
#     test_arrays[prev_count + i] = model.docvecs[prefix_test_neu]
#     test_labels[prev_count + i] = 2


print "Begin classification..."
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

print "Accuracy:", classifier.score(test_arrays, test_labels)
