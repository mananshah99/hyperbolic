from __future__ import print_function
import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import *
from sklearn.preprocessing import MultiLabelBinarizer
from time import time
from sklearn.linear_model import LogisticRegression

EMBEDDINGS_FILE = '../../graphsage/embeddings/graphsage_email-Eu-core.txt' #'../poincare/embeddings/poincare_email_burn.txt'
LABELS_FILE = '../../data/email/email-Eu-core-department-labels.txt'
HEADER = False

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["f1-micro", "f1-macro", "f1-samples", "f1-weighted"]

        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average[3:])
        results["accuracy"] = accuracy_score(Y, Y_)
        
        print('-------------------')
        print(results)
        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def load_embeddings(filename):
    fin = open(filename, 'r')
    if HEADER:
        node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        if HEADER:
            assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    if HEADER:
        assert len(vectors) == node_num
    return vectors


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


vectors = load_embeddings(EMBEDDINGS_FILE)
X, Y = read_node_label(LABELS_FILE)
print("Training classifier using {:.2f}% nodes...".format(0.8*100))
clf = Classifier(vectors=vectors, clf=LogisticRegression())
clf.split_train_evaluate(X, Y, 0.8)
