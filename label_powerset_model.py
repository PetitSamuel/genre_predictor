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
from util import BASE_GENRES, parse_data_multi_output_full, parse_data_single_output_full
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

tensValues = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
wholeValues = [1, 10, 100, 1000]
modelLabels = []
modelScores = []
figureNo = 0
splitNo = 10


def main():
    with open('./data/11k_songs_tso_dataset.json') as f:
        data = json.load(f)

    # max_genres = 2
    # x, y = parse_data_multi_output_full(data, max_genres)
    x, y = parse_data_single_output_full(data, True)

    bernoulliNB(x, y)
    decisionTreeClassifier(x, y)
    extraTreeClassifier(x, y)
    extraTreesClassifier(x, y)
    gaussianNB(x, y)
    kNeighborsClassifier(x, y)
    labelPropagation(x, y)
    labelSpreading(x, y)
    linearDiscriminantAnalysis(x, y)
    linearSVC(x, y)
    svc(x, y)
    logisticRegression(x, y)
    mlpClassifier(x, y)
    nearestCentroid(x, y)
    randomForestClassifier(x, y)
    ridgeClassifier(x, y)
    ridgeClassifierCV(x, y)
    dummyClassifier(x, y)

    comparisonGraph()
    showGraphs()


# multi - 2 genres - 14% # single - 35%
def bernoulliNB(x, y):
    model = BernoulliNB()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 47% # single - 58%
def decisionTreeClassifier(x, y):
    model = DecisionTreeClassifier()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 48% # single - 56%
def extraTreeClassifier(x, y):
    model = ExtraTreeClassifier()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 51% # single - 64%
def extraTreesClassifier(x, y):
    model = ExtraTreesClassifier()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres -  6% # single - 23%
def gaussianNB(x, y):
    model = GaussianNB()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 28% # single - 41%
def kNeighborsClassifier(x, y):
    scores = []
    mses = []
    for k in wholeValues:
        model = KNeighborsClassifier(n_neighbors=k)
        accuracy, mse = runSet(model, x, y)
        scores.append(accuracy)
        mses.append(mse)
    showMSEGraph(wholeValues, scores, mses, "k", model.__class__.__name__)
    addModelComparison(model, max(scores))


# multi - 2 genres - 29% # single - 58%
def labelPropagation(x, y):
    model = LabelPropagation()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 30% # single - 56%
def labelSpreading(x, y):
    model = LabelSpreading()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 27% # single - 42%
def linearDiscriminantAnalysis(x, y):
    model = LinearDiscriminantAnalysis()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 30% # single - 41%
def linearSVC(x, y):
    scores = []
    mses = []
    for C in tensValues:
        model = LinearSVC(max_iter=10000, C=C)
        accuracy, mse = runSet(model, x, y)
        scores.append(accuracy)
        mses.append(mse)
    showMSEGraph(tensValues, scores, mses, "C", model.__class__.__name__)
    addModelComparison(model, max(scores))


# multi - 2 genres - 30% # single - 43%
def svc(x, y):
    scores = []
    mses = []
    for C in tensValues:
        model = SVC(C=C)
        accuracy, mse = runSet(model, x, y)
        scores.append(accuracy)
        mses.append(mse)
    showMSEGraph(tensValues, scores, mses, "C", model.__class__.__name__)
    addModelComparison(model, max(scores))


# multi - 2 genres - 30% # single - 44%
def logisticRegression(x, y):
    scores = []
    mses = []
    for C in tensValues:
        model = LogisticRegression(max_iter=500, C=C)
        accuracy, mse = runSet(model, x, y)
        scores.append(accuracy)
        mses.append(mse)
    showMSEGraph(tensValues, scores, mses, "C", model.__class__.__name__)
    addModelComparison(model, max(scores))


# multi - 2 genres - 31% # single - 47%
def mlpClassifier(x, y):
    model = MLPClassifier(max_iter=1000)
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres -  7% # single - 31%
def nearestCentroid(x, y):
    model = NearestCentroid()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 51% # single - 64%
def randomForestClassifier(x, y):
    model = RandomForestClassifier()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 28% # single - 38%
def ridgeClassifier(x, y):
    scores = []
    mses = []
    for alpha in tensValues:
        model = RidgeClassifier(alpha=alpha)
        accuracy, mse = runSet(model, x, y)
        scores.append(accuracy)
        mses.append(mse)
    showMSEGraph(tensValues, scores, mses, "alpha", model.__class__.__name__)
    addModelComparison(model, max(scores))


# multi - 2 genres - 28% # single - 37%
def ridgeClassifierCV(x, y):
    model = RidgeClassifierCV()
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


# multi - 2 genres - 14% # single - 29%
def dummyClassifier(x, y):
    model = DummyClassifier(strategy="most_frequent")
    accuracy, mse = runSet(model, x, y)
    # printSet(model, accuracy, mse)
    addModelComparison(model, accuracy)


def runSet(model, x, y):
    mse = []
    accuracy = []
    kf = KFold(n_splits=splitNo)
    for train, test in kf.split(x):
        classifier = LabelPowerset(model)
        classifier.fit(x[train], y[train])
        predictions = classifier.predict(x[test])
        accuracy.append(accuracy_score(y[test], predictions))
        mse.append(mean_squared_error(y[test], predictions.toarray()))
    mse = np.array(mse)
    accuracy = np.array(accuracy)

    mse = np.mean(mse)
    accuracy = np.mean(accuracy)

    return accuracy, mse


def printSet(model, accuracy, mse):
    print(
        model.__class__.__name__,
        "\n\tAccuracy:\t\t", accuracy,
        "\n\tMean Squared Error:\t", mse
    )


def showMSEGraph(values, scores, mses, xName, modelName):
    plt.figure(getFigureNo())
    plt.rc("font", size=16)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(values, scores, yerr=mses, color="black", linewidth=2)
    plt.xlabel(xName)
    plt.ylabel("Mean")
    plt.legend([modelName])


def comparisonGraph():
    plt.figure(getFigureNo())
    plt.rc("font", size=16)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.bar(modelLabels, modelScores, color='green')
    plt.xlabel("Model")
    plt.ylabel("Accuracy")


def showGraphs():
    plt.show()


def addModelComparison(model, score):
    modelLabels.append(model.__class__.__name__)
    modelScores.append(score)


def getFigureNo():
    global figureNo
    figureNo += 1
    return figureNo


main()
