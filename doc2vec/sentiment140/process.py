from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from sklearn.linear_model import LogisticRegression


model = Doc2Vec.load('./sentiment140.d2v')

# train_pos_count = 898
# train_neg_count = 887
# test_pos_count = 100
# test_neg_count = 98

train_pos_count = 2290
train_neg_count = 848
test_pos_count = 342
test_neg_count = 175

train_arrays = numpy.zeros((train_pos_count + train_neg_count, 50))
train_labels = numpy.zeros(train_pos_count + train_neg_count)

for i in range(train_pos_count):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in range(train_neg_count):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[train_pos_count + i] = model.docvecs[prefix_train_neg]
    train_labels[train_pos_count + i] = 0


test_arrays = numpy.zeros((test_pos_count + test_neg_count, 50))
test_labels = numpy.zeros(test_pos_count + test_neg_count)

for i in range(test_pos_count):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_labels[i] = 1

for i in range(test_neg_count):
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[test_pos_count + i] = model.docvecs[prefix_test_neg]
    test_labels[test_pos_count + i] = 0


classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

print classifier.score(test_arrays, test_labels)
