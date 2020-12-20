import pandas as pd
import numpy as np
import json

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import accuracy_score
from util import BASE_GENRES


def main():
    with open('./data/11k_songs_tso_dataset.json') as f:
        data = json.load(f)

    max_genres = 2
    x_train, y_train, x_hold_out, y_hold_out = parse_data_multi_output(
        data, max_genres)

    # model = BernoulliNB()                       # 14%
    # model = DecisionTreeClassifier()            # 47%
    # model = ExtraTreeClassifier()               # 48%
    # model = ExtraTreesClassifier()              # 51%
    # model = GaussianNB()                        # 6%
    # model = KNeighborsClassifier()              # 28%
    # model = LabelPropagation()                  # 29%
    # model = LabelSpreading()                    # 30%
    # model = LinearDiscriminantAnalysis()        # 27%
    # model = LinearSVC()                         # 30%
    # model = LogisticRegression(max_iter=500)    # 30%
    # model = LogisticRegressionCV(max_iter=1000) #
    # model = MLPClassifier()                     # 31%
    # model = NearestCentroid()                   # 7%
    # model = QuadraticDiscriminantAnalysis()     #
    # model = RadiusNeighborsClassifier()         # 16%
    # model = RandomForestClassifier()            # 51%
    # model = RidgeClassifier()                   # 28%
    model = RidgeClassifierCV()                 # 28%

    classifier = LabelPowerset(model)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_hold_out)
    print("Accuracy = ", accuracy_score(y_hold_out, predictions))


def parse_data_multi_output(data, max):
    x = []
    y = []
    for i in range(len(data)):
        if not data[i]['genres']:
            continue
        else:
            labels = data[i]['genres'][:max]

        if labels[0] in ['electro']:
            continue

        features = [
            data[i]['danceability'],
            data[i]['energy'],
            data[i]['speechiness'],
            data[i]['acousticness'],
            data[i]['instrumentalness'],
            data[i]['liveness'],
            data[i]['valence'],
        ]
        x.append(features)
        y.append(labels)

    x = np.array(x)
    y = map_true_false(y)
    y = np.array(y)

    from sklearn.utils import shuffle
    x, y = shuffle(x, y)
    x_train = x[:10000]
    y_train = y[:10000]
    x_hold_out = x[10000:]
    y_hold_out = y[10000:]

    return x_train, y_train, x_hold_out, y_hold_out


def map_true_false(y):
    mapped_y = []
    for curr_song_genres in y:
        song_list = []
        for g in BASE_GENRES:
            if g in curr_song_genres:
                song_list.append(True)
            else:
                song_list.append(False)
        mapped_y.append(song_list)
    return mapped_y


main()
