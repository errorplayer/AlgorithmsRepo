from sklearn.datasets import *
from sklearn import tree
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import pickle

def safe_equal(a, b):
    if abs(a-b) < 0.000001:
        return True
    return False

def loadData(fileName):
    data=pd.read_csv(fileName,header=None)
    data=data.values
    y_label=data[:,0]
    x_label=data[:,1:]
    return x_label,y_label

def t(a):
    print(type(a))
    exit(-10000)

def l(a):
    print(len(a))
    exit(-10000)

def s(a):
    print(a.shape)
    exit(-10000)

def p(a):
    print(a)
    exit(-10000)

clear_data_home()
# X, y = load_wine(return_X_y=True)
# X, y = load_breast_cancer(return_X_y=True)

# X, y = load_digits(return_X_y=True)
X, y = load_iris(return_X_y=True)


def getboundary(nums):
    max_ = float('-inf')
    min_ = float('inf')
    for i in nums:
        if i > max_:
            max_ = i
        if i < min_:
            min_ = i
    return min_, max_

def calc_gini(indice, samples, labels, classes):
    """
    A function for calculating the Gini value of a set of samples.

    :param indice:  indicating the elements belonging to this sample set
    :param samples: total training samples
    :param labels: total labels of training samples
    :param classes: the list of classes
    :return gini: the Gini value
    """
    l_ = labels[indice]
    size = len(indice)
    gini = 1
    substract_ = 0
    for i in classes:
        tmp_size = len(l_[l_ == i])
        substract_ += tmp_size ** 2

    return gini - (substract_ / (size ** 2))

def calc_gini_A(indice, samples, labels, classes, index_A):
    """
    A function for calculating the Gini value of the set of samples given splitting feature index.

    :param indice:  indicating the elements belonging to this sample set
    :param samples: total training samples
    :param labels: total labels of training samples
    :param classes: the list of classes
    :param index_A: the splitting feature index
    :return gini: the Gini value
    """

    s_ = samples[indice]
    l_ = labels[indice]
    s_ = np.array(s_)
    l_ = np.array(l_)

    size  = len(indice)
    feature_A_samples = s_[:, index_A]
    feature_A_samples = feature_A_samples.flatten()
    feature_A_labels = l_

    maximum, minimum = getboundary(feature_A_samples)
    splitting_feature_value = (maximum + minimum) / 2
    if safe_equal(maximum, minimum):
        return 1.0, splitting_feature_value



    index_D1 = np.where(feature_A_samples <= splitting_feature_value)
    index_D2 = np.where(feature_A_samples > splitting_feature_value)

    D1_samples = feature_A_samples[index_D1]
    D2_samples = feature_A_samples[index_D2]
    D1_labels = feature_A_labels[index_D1]
    D2_labels = feature_A_labels[index_D2]
    D1_size = len(index_D1[0])
    D2_size = len(index_D2[0])
    if D1_size == 0 or D2_size == 0:
        return 0.0, splitting_feature_value
    indice_d1 = [i for i in range(D1_size)]
    indice_d2 = [i for i in range(D2_size)]
    gini = (D1_size/size) * calc_gini(indice_d1, D1_samples, D1_labels, classes) + (D2_size/size) * calc_gini(indice_d2, D2_samples, D2_labels, classes)
    return gini, splitting_feature_value

def split_data(splitting_feature_idx, splitting_value, samples, labels):
    size1 = len(samples)
    size2 = len(labels)
    assert size1==size2

    feature_A_samples = samples[:, splitting_feature_idx]
    feature_A_samples = feature_A_samples.flatten()
    feature_A_labels = labels

    index_D1 = np.where(feature_A_samples <= splitting_value)
    index_D2 = np.where(feature_A_samples > splitting_value)

    D1_samples = samples[index_D1]
    D2_samples = samples[index_D2]
    D1_labels = labels[index_D1]
    D2_labels = labels[index_D2]

    return D1_samples, D2_samples, D1_labels, D2_labels

def getLargestClass(x_train, y_train):
    d = {}
    for i in y_train:
        try:
            d[i] += 1
        except:
            d[i] = 1
    max_count = 0
    max_k = 0
    for k in d.keys():
        if d[k] > max_count:
            max_count = d[k]
            max_k = k
    size = len(y_train)
    return max_k, max_count/size

def createTree(x_train, y_train):
    classes = set([i for i in y_train])
    if len(classes) == 1:
        return y_train[0]
    max_feature, proba = getLargestClass(x_train, y_train)
    if proba > 0.85:
        return max_feature
    size = len(x_train)
    feature_dim = len(x_train[0])
    idx = [i for i in range(size)]
    mini_gini = float('inf')
    splitting_value = 0
    splitting_feature_idx = 0
    for i in range(feature_dim):
        t_gini, tmp_split_value = calc_gini_A(idx, x_train, y_train, classes, [i])
        if t_gini < mini_gini:
            mini_gini = t_gini
            splitting_value = tmp_split_value
            splitting_feature_idx = i

    d1_s, d2_s, d1_l, d2_l = split_data(splitting_feature_idx, splitting_value, x_train, y_train)
    if len(d2_s) == 0:
        max_feature, proba = getLargestClass(d1_s, d1_l)
        return max_feature
    if len(d1_s) == 0:
        max_feature, proba = getLargestClass(d2_s, d2_l)
        return max_feature
    tree_dict = {splitting_feature_idx:{}}

    tree_dict[splitting_feature_idx][0]= createTree(d1_s, d1_l)
    tree_dict[splitting_feature_idx][1]= createTree(d2_s, d2_l)
    tree_dict[splitting_feature_idx][2]= splitting_value
    return tree_dict

def predict(x_test, tree_dict):
    result = []
    for i in x_test:
        try:
            (key, value), = tree_dict.items()
        except:
            result.append(tree_dict)
            continue

        tree = {}
        while type(value).__name__=='dict':
                if i[int(key)] <= value[2]:
                    tree = value[0]
                else:
                    tree = value[1]
                if type(tree).__name__=='int32' or type(tree).__name__=='int64':
                    result.append(tree)
                    break
                (key, value), = tree.items()

    return result

def test(y_true, y_test):
    acc_count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_test[i]:
            acc_count += 1
    size1 = len(y_true)
    size2 = len(y_test)
    assert size1 == size2
    print(round(acc_count/size1, 5))

X, y = shuffle(X, y)
data_size = len(X)
cut_ = int(data_size*(3/4))
x_train = X[:cut_]
y_train = y[:cut_]
x_test = X[cut_:]
y_test = y[cut_:]

x_train, y_train = loadData('./mnist/mnist_train.csv')
x_test, y_test = loadData('./mnist/mnist_test.csv')
x_train, y_train = shuffle(x_train, y_train)

print('number of the training data is {}'.format(x_train.shape[0]))
print('number of the test data is {}'.format(x_test.shape[0]))
print('dim of the training data is {}'.format(x_train.shape[1]))
print('----------------------------------my code results:----------------------------------------')
cart = createTree(x_train, y_train)
y_predicted = predict(x_test, cart)
test(y_test, y_predicted)
f = open('./mnist/mnist_classifier_cart', 'wb')
pickle.dump(cart, f)
f.close()
# print(cart)
print('----------------------------------scikit-learn lib results:----------------------------------------')
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
print(round(clf.score(x_test, y_test), 5))








