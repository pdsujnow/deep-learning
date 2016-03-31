import numpy
from gensim.models import Doc2Vec
from six.moves import cPickle

model = Doc2Vec.load('../doc2vec/sentiment140/sentiment.d2v')
train_sents = []
train_labels = []
test_sents = []
test_labels = []

train_pos_count = 98912
train_neg_count = 99309
test_pos_count = 182
test_neg_count = 177

for i in range(train_pos_count):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_sents.append(model.docvecs[prefix_train_pos].tolist())
    train_labels.append(1)

for i in range(train_neg_count):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_sents.append(model.docvecs[prefix_train_neg].tolist())
    train_labels.append(0)


for i in range(test_pos_count):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_sents.append(model.docvecs[prefix_test_pos].tolist())
    test_labels.append(1)

for i in range(test_neg_count):
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_sents.append(model.docvecs[prefix_test_neg].tolist())
    test_labels.append(0)


train_tuple = (train_sents, train_labels)
test_tuple = (test_sents, test_labels)

f = open('tweets.pkl', 'wb')
cPickle.dump(train_tuple, f)
cPickle.dump(test_tuple, f)
f.close()
