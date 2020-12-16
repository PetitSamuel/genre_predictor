import pandas as pd
import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


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
            data[i]['valence'],
        ]
        x.append(features)
        y.append(labels)

    y = np.array(y)
    x = np.array(x)

    from sklearn.utils import shuffle
    x, y = shuffle(x, y)
    x_train = x[:10000]
    y_train = y[:10000]
    x_hold_out = x[10000:]
    y_hold_out = y[10000:]
    print(len(x_hold_out))

    # train and test each model - accuracy %
    # knn(x, y)               # 40%
    # decision_tree(x, y)  # 56%
    # random_forest(x, y)     # 61+%
    # baseline(x,y)           # 25%
    # logistic_reg(x,y.ravel()) # 40%
    # lr_cv_q(x, y)
    # lr_cv_C(x,y)
    # knn_cv_n_neighbours(x,y,[1,2,3]) # k = 1
    # knn_cv_gamma(x,y,[0.1, 1, 10, 100]) # with k=1 gaussian kernel doesn't have much effect
    # about 55%

    knn_hold_out(x_train, y_train, x_hold_out, y_hold_out)


def lr_cv_q(x, y):
    from sklearn.utils import shuffle
    x, y, = shuffle(x, y)

    mean_error = []
    std_error = []
    q_range = [1, 2, 3]  # seems q=2 is slightly better than q=1
    for q in q_range:
        from sklearn.preprocessing import PolynomialFeatures
        xpoly = PolynomialFeatures(q).fit_transform(x)

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(multi_class='ovr', solver='liblinear', C=100)
        temp = []

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        for train, test in kf.split(xpoly):
            model.fit(xpoly[train], y[train].ravel())

            score = model.score(xpoly[test], y[test])
            print(score)
            temp.append(score)
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.errorbar(q_range, mean_error, yerr=std_error, elinewidth=3)
    plt.xlabel('q')
    plt.ylabel('Mean accuracy')
    plt.show()


def lr_cv_C(x, y):
    mean_error = []
    std_error = []
    C_range = [5, 8, 10, 20, 50]
    for C in C_range:

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(multi_class='ovr', solver='liblinear', C=C)
        temp = []
        plotted = False

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        for train, test in kf.split(x):
            model.fit(x[train], y[train])

            score = model.score(x[test], y[test])
            print(score)
            temp.append(score)
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.errorbar(C_range, mean_error, yerr=std_error, elinewidth=3)
    plt.xlabel('C')
    plt.ylabel('Mean accuracy')
    plt.show()


def logistic_reg(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    from sklearn.linear_model import LogisticRegression
    lr_pipeline = Pipeline([
        ('lr', LogisticRegression(multi_class='ovr', solver='liblinear'))
    ])

    lr_params = {
        'lr__C': [1000, 100, 10]
    }

    clf_lr = GridSearchCV(lr_pipeline, lr_params)

    # fit on the best hyperparameters
    clf_lr.fit(x_train, y_train.ravel())

    print("Best Score: ", clf_lr.best_score_)
    print("Best Params: ", clf_lr.best_params_)

    print(clf_lr.score(x_test, y_test))

    show_confusion_matrix(clf_lr, x_test, y_test)


# baseline - most frequent - "pop"
def baseline(x, y):
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x, y)

    y_pred = dummy.predict(x)

    from sklearn.metrics import classification_report
    print(classification_report(y_pred, y))


def create_gaussian_kernel(gamma):
    def g(distances):
        weights = np.exp(-gamma * (distances ** 2))
        return weights / np.sum(weights)

    return g


def knn_cv_n_neighbours(x, y, neighbours):
    mean_error = []
    std_error = []

    for neighbour in neighbours:
        # kernel = create_gaussian_kernel(gamma)
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=neighbour)

        temp = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        for train, test in kf.split(x):
            model.fit(x[train], y[train])

            score = model.score(x[test], y[test])
            print(score)
            temp.append(score)
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.errorbar(neighbours, mean_error, yerr=std_error)
    plt.title("#Neighbours vs Mean Accuracy with standard error")
    plt.xlabel("# Neighbours")
    plt.ylabel("Mean accuracy")
    plt.show()


def knn_cv_gamma(x, y, gammas):
    mean_error = []
    std_error = []

    for g in gammas:
        kernel = create_gaussian_kernel(g)
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=1, weights=kernel)

        temp = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        for train, test in kf.split(x):
            model.fit(x[train], y[train])

            score = model.score(x[test], y[test])
            print(score)
            temp.append(score)
        print(np.array(temp))
        print("mean ", np.array(temp).mean())
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    print(mean_error)
    print(std_error)
    plt.errorbar(gammas, mean_error, yerr=std_error)
    plt.title("gamma vs Mean Accuracy with standard error")
    plt.xlabel("gamma")
    plt.ylabel("Mean accuracy")
    plt.show()


def knn_hold_out(x, y, x_hold, y_hold):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=1, weights="uniform")
    model.fit(x, y.ravel())
    print("hold-out score", model.score(x_hold, y_hold))


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

    show_confusion_matrix(clf_knn, x_test, y_test)


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

    show_confusion_matrix(clf_rf, x_test, y_test)


# haven't looked at hyperparameters for this yet
def decision_tree(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    clf_dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    clf_dt.fit(x_train, y_train)

    y_pred = clf_dt.predict(x_test)

    from sklearn.metrics import classification_report
    print(classification_report(y_pred, y_test))

    show_confusion_matrix(clf_dt, x_test, y_test)


def show_confusion_matrix(clf, x, y):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Confusion Matrx')

    from sklearn.metrics import plot_confusion_matrix
    plot_confusion_matrix(clf, x, y, ax=ax)
    plt.xticks(rotation=35)
    plt.show()


main()
