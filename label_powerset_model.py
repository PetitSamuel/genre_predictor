import pandas as pd
import numpy as np
import json

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from util import BASE_GENRES, parse_data_multi_output, parse_data_single_output


def main():
    with open('./data/11k_songs_tso_dataset.json') as f:
        data = json.load(f)

    # max_genres = 2
    # x_train, y_train, x_hold_out, y_hold_out = parse_data_multi_output(
    #     data, max_genres)
    x_train, y_train, x_hold_out, y_hold_out = parse_data_single_output(
        data, True)

    bernoulliNB(x_train, y_train, x_hold_out, y_hold_out)
    decisionTreeClassifier(x_train, y_train, x_hold_out, y_hold_out)
    extraTreeClassifier(x_train, y_train, x_hold_out, y_hold_out)
    extraTreesClassifier(x_train, y_train, x_hold_out, y_hold_out)
    gaussianNB(x_train, y_train, x_hold_out, y_hold_out)
    kNeighborsClassifier(x_train, y_train, x_hold_out, y_hold_out)
    labelPropagation(x_train, y_train, x_hold_out, y_hold_out)
    labelSpreading(x_train, y_train, x_hold_out, y_hold_out)
    linearDiscriminantAnalysis(x_train, y_train, x_hold_out, y_hold_out)
    linearSVC(x_train, y_train, x_hold_out, y_hold_out)
    svc(x_train, y_train, x_hold_out, y_hold_out)
    logisticRegression(x_train, y_train, x_hold_out, y_hold_out)
    mlpClassifier(x_train, y_train, x_hold_out, y_hold_out)
    nearestCentroid(x_train, y_train, x_hold_out, y_hold_out)
    randomForestClassifier(x_train, y_train, x_hold_out, y_hold_out)
    ridgeClassifier(x_train, y_train, x_hold_out, y_hold_out)
    ridgeClassifierCV(x_train, y_train, x_hold_out, y_hold_out)
    dummyClassifier(x_train, y_train, x_hold_out, y_hold_out)


# multi - 2 genres - 14% # single - 35%
def bernoulliNB(x_train, y_train, x_hold_out, y_hold_out):
    model = BernoulliNB()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 47% # single - 58%
def decisionTreeClassifier(x_train, y_train, x_hold_out, y_hold_out):
    model = DecisionTreeClassifier()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 48% # single - 56%
def extraTreeClassifier(x_train, y_train, x_hold_out, y_hold_out):
    model = ExtraTreeClassifier()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 51% # single - 64%
def extraTreesClassifier(x_train, y_train, x_hold_out, y_hold_out):
    model = ExtraTreesClassifier()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres -  6% # single - 23%
def gaussianNB(x_train, y_train, x_hold_out, y_hold_out):
    model = GaussianNB()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 28% # single - 41%
def kNeighborsClassifier(x_train, y_train, x_hold_out, y_hold_out):
    model = KNeighborsClassifier()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 29% # single - 58%
def labelPropagation(x_train, y_train, x_hold_out, y_hold_out):
    model = LabelPropagation()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 30% # single - 56%
def labelSpreading(x_train, y_train, x_hold_out, y_hold_out):
    model = LabelSpreading()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 27% # single - 42%
def linearDiscriminantAnalysis(x_train, y_train, x_hold_out, y_hold_out):
    model = LinearDiscriminantAnalysis()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 30% # single - 41%
def linearSVC(x_train, y_train, x_hold_out, y_hold_out):
    model = LinearSVC(max_iter=10000)
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 30% # single - 43%
def svc(x_train, y_train, x_hold_out, y_hold_out):
    model = LogisticRegression(max_iter=500)
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 30% # single - 44%
def logisticRegression(x_train, y_train, x_hold_out, y_hold_out):
    model = LogisticRegression(max_iter=500)
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 31% # single - 47%
def mlpClassifier(x_train, y_train, x_hold_out, y_hold_out):
    model = MLPClassifier(max_iter=1000)
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres -  7% # single - 31%
def nearestCentroid(x_train, y_train, x_hold_out, y_hold_out):
    model = NearestCentroid()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 51% # single - 64%
def randomForestClassifier(x_train, y_train, x_hold_out, y_hold_out):
    model = RandomForestClassifier()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 28% # single - 38%
def ridgeClassifier(x_train, y_train, x_hold_out, y_hold_out):
    model = RidgeClassifier()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 28% # single - 37%
def ridgeClassifierCV(x_train, y_train, x_hold_out, y_hold_out):
    model = RidgeClassifierCV()
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


# multi - 2 genres - 14% # single - 29%
def dummyClassifier(x_train, y_train, x_hold_out, y_hold_out):
    model = DummyClassifier(strategy="most_frequent")
    accuracy, mse = runSet(model, x_train, y_train, x_hold_out, y_hold_out)
    printSet(model, accuracy, mse)


def runSet(model, x_train, y_train, x_hold_out, y_hold_out):
    classifier = LabelPowerset(model).fit(x_train, y_train)
    predictions = classifier.predict(x_hold_out)
    accuracy = accuracy_score(y_hold_out, predictions)
    mse = mean_squared_error(y_hold_out, predictions.toarray())
    return accuracy, mse


def printSet(model, accuracy, mse):
    print(
        model.__class__.__name__,
        "\n\tAccuracy:\t\t", accuracy,
        "\n\tMean Squared Error:\t", mse
    )


main()
