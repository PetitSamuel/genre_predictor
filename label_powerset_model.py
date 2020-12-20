import pandas as pd
import numpy as np
import json

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from util import BASE_GENRES, parse_data_multi_output, parse_data_single_output


def main():
    with open('./data/11k_songs_tso_dataset.json') as f:
        data = json.load(f)

    # max_genres = 2
    # x_train, y_train, x_hold_out, y_hold_out = parse_data_multi_output(
    #     data, max_genres)
    x_train, y_train, x_hold_out, y_hold_out = parse_data_single_output(
        data, True)

    model = BernoulliNB()
    # multi - 2 genres - 14% # single - 35%
    # model = DecisionTreeClassifier()
    # # multi - 2 genres - 47% # single - 58%
    # model = ExtraTreeClassifier()
    # # multi - 2 genres - 48% # single - 56%
    # model = ExtraTreesClassifier()
    # # multi - 2 genres - 51% # single - 64%
    # model = GaussianNB()
    # # multi - 2 genres -  6% # single - 23%
    # model = KNeighborsClassifier()
    # # multi - 2 genres - 28% # single - 41%
    # model = LabelPropagation()
    # # multi - 2 genres - 29% # single - 58%
    # model = LabelSpreading()
    # # multi - 2 genres - 30% # single - 56%
    # model = LinearDiscriminantAnalysis()
    # # multi - 2 genres - 27% # single - 42%
    # model = LinearSVC(max_iter=10000)
    # # multi - 2 genres - 30% # single - 41%
    # model = LogisticRegression(max_iter=500)
    # # multi - 2 genres - 30% # single - 44%
    # model = MLPClassifier(max_iter=1000)
    # # multi - 2 genres - 31% # single - 47%
    # model = NearestCentroid()
    # # multi - 2 genres -  7% # single - 31%
    # model = RandomForestClassifier()
    # # multi - 2 genres - 51% # single - 64%
    # model = RidgeClassifier()
    # # multi - 2 genres - 28% # single - 38%
    # model = RidgeClassifierCV()
    # # multi - 2 genres - 28% # single - 37%
    # model = DummyClassifier(strategy="most_frequent")
    # # multi - 2 genres - 14% # single - 29%

    classifier = LabelPowerset(model)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_hold_out)
    print("Accuracy = ", accuracy_score(y_hold_out, predictions))


main()
