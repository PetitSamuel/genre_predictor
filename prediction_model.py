import pandas as pd
import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Slightly simplified the problem to just picking the most likely genre given a set of features

def main():
    with open('./data/11k_songs_tso_dataset.json') as f:
        data = json.load(f)
    print(data[0])

    # Splitting data into inputs and labels
    x = []
    y = []
    for i in range(len(data)):
        if not data[i]['genres']:
            continue
        if len(data[i]['genres']) == 1:
            labels = data[i]['genres']
        else:
            labels = data[i]['genres'][:1]

        features = [
            data[i]['danceability'],
            data[i]['energy'],
            data[i]['speechiness'],
            data[i]['acousticness'],
            data[i]['instrumentalness'],
            data[i]['liveness'],
            data[i]['valence']
        ]
        x.append(features)
        y.append(labels)

    y = np.array(y)
    x = np.array(x)

    # train and test each model - accuracy %
    # knn(x, y)               # 40%
    # decision_tree(x,y)      # 56%
    # random_forest(x, y)     # 61%
    baseline(x,y)           # 25%


# baseline - most frequent - "pop"
def baseline(x,y):
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x,y)

    y_pred = dummy.predict(x)

    from sklearn.metrics import classification_report
    print(classification_report(y_pred, y))


# knn network - number of neighbours cross validated
# though non deterministic number chosen each time
# with no real change in accuracy results
def knn(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    knn_pipeline = Pipeline([
        ('knn', KNeighborsClassifier())
    ])

    # hyperparameters to train
    grid_params = {
        'knn__n_neighbors': [39, 41, 43, 45, 47, 51, 71]
    }
    clf_knn = GridSearchCV(knn_pipeline, grid_params)

    # fit on the best hyperparameters
    clf_knn.fit(x_train, y_train.ravel())

    print("Best Score: ", clf_knn.best_score_)
    print("Best Params: ", clf_knn.best_params_)

    # classification test
    y_pred = clf_knn.predict(x_test)
    from sklearn.metrics import classification_report
    print(classification_report(y_pred, y_test))


# random forest classifier
# n_estimators chosen through cross validation - takes a while
# so left at 200
def random_forest(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    rf_pipeline = Pipeline([
        ('rfor', RandomForestClassifier(random_state=42))
    ])

    # hyperparameters to train
    rf_params = {'rfor__n_estimators': [200]}
    clf_rf = GridSearchCV(rf_pipeline, rf_params)

    # fit on the best hyperparameters
    clf_rf.fit(x_train, y_train.ravel())

    print("Best score: ", clf_rf.best_score_)
    print("Best params: ", clf_rf.best_params_)

    # classification test
    y_pred = clf_rf.predict(x_test)
    from sklearn.metrics import classification_report
    print(classification_report(y_pred, y_test))


# haven't looked at hyperparameters for this yet
def decision_tree(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    clf_dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    clf_dt.fit(x_train, y_train)

    y_pred = clf_dt.predict(x_test)

    from sklearn.metrics import classification_report
    print(classification_report(y_pred, y_test))


main()
