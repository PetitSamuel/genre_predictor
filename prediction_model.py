import pandas as pd
import numpy as np
import json

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from util import BASE_GENRES


# Slightly simplified the problem to just picking the most likely genre given a set of features

def main():
    with open('./data/11k_songs_tso_dataset.json') as f:
        data = json.load(f)
    # for single output models
    # x_train,y_train,x_hold_out,y_hold_out = parse_data_single_output(data)

    # for multi output models
    max_genres = 2
    x_train, y_train, x_hold_out, y_hold_out = parse_data_multi_output(
        data, max_genres)

    # train and test each model - accuracy %
    # knn(x_train, y_train)               # 40%
    # decision_tree(x_train, y_train)  # 56%
    # random_forest(x_train, y_train)     # 61+%
    # baseline(x_train,y_train)           # 25%
    # logistic_reg(x_train,y_train.ravel()) # 40%
    # lr_cv_q(x_train, y_train)
    # lr_cv_C(x_train,y_train)
    # knn_cv_n_neighbours(x_train,y_train,[1,2,3]) # k = 1
    # knn_cv_gamma(x_train,y_train,[0.1, 1, 10, 100]) # with k=1 gaussian kernel doesn't have much effect
    # about 55%

    # knn_hold_out(x_train, y_train, x_hold_out, y_hold_out)

    # best_model = chain_classifiers(x_train, y_train)
    # chain_of_classifiers_hold_out(x_hold_out, y_hold_out, best_model, max_genres)

    knn_multi_output(x_train, y_train)


# print a few predictions on the held out dataset
def chain_of_classifiers_hold_out(x, y, model, max_genres):
    preds = model.predict(x)
    for i in range(50):
        print("Input", true_false_to_genres(x[i])[:max_genres])
        print("Pred", true_false_to_genres(preds[i]))
        print()


# convert True/False array to Genres
def true_false_to_genres(tf_arr):
    genres = []
    for i in range(len(tf_arr)):
        if tf_arr[i]:
            genres.append(BASE_GENRES[i])
    return genres


# Map array of genres to True/False values for multi output models
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


# max = max labels per datum
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


def parse_data_single_output(data):
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

        if labels[0] in ['electro']:
            print("skipping electro")
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

    y = np.array(y)
    x = np.array(x)

    from sklearn.utils import shuffle
    x, y = shuffle(x, y)
    x_train = x[:10000]
    y_train = y[:10000]
    x_hold_out = x[10000:]
    y_hold_out = y[10000:]

    return x_train, y_train, x_hold_out, y_hold_out


# chain of classifers example from
# https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html#sphx-glr-auto-examples-multioutput-plot-classifier-chain-yeast-py
def chain_classifiers(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    base_lr = LogisticRegression()
    ovr = OneVsRestClassifier(base_lr)
    ovr.fit(x_train, y_train)
    y_pred_ovr = ovr.predict(x_test)

    from sklearn.metrics import jaccard_score
    ovr_jaccard_score = jaccard_score(y_test, y_pred_ovr, average='samples')

    from sklearn.multioutput import ClassifierChain
    chains = [ClassifierChain(base_lr, order='random', random_state=i)
              for i in range(10)]

    for chain in chains:
        chain.fit(x_train, y_train)

    y_pred_chains = np.array([chain.predict(x_test) for chain in
                              chains])
    chain_jaccard_scores = [jaccard_score(y_test, y_pred_chain >= 0.5,
                                          average='samples')
                            for y_pred_chain in y_pred_chains]

    y_pred_ensemble = y_pred_chains.mean(axis=0)
    ensemble_jaccard_score = jaccard_score(y_test,
                                           y_pred_ensemble >= 0.5,
                                           average='samples')

    model_scores = [ovr_jaccard_score] + chain_jaccard_scores
    model_scores.append(ensemble_jaccard_score)

    model_names = ('Independent',
                   'Chain 1',
                   'Chain 2',
                   'Chain 3',
                   'Chain 4',
                   'Chain 5',
                   'Chain 6',
                   'Chain 7',
                   'Chain 8',
                   'Chain 9',
                   'Chain 10',
                   'Ensemble')

    x_pos = np.arange(len(model_names))

    # Plot the Jaccard similarity scores for the independent model, each of the
    # chains, and the ensemble (note that the vertical axis on this plot does
    # not begin at 0).

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid(True)
    ax.set_title('Classifier Chain Ensemble Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation='vertical')
    ax.set_ylabel('Jaccard Similarity Score')
    ax.set_ylim([min(model_scores) * .9, max(model_scores) * 1.1])
    colors = ['r'] + ['b'] * len(chain_jaccard_scores) + ['g']
    ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
    plt.tight_layout()
    plt.show()

    return chains[-1]


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
        model = LogisticRegression(
            multi_class='ovr', solver='liblinear', C=100)
        temp = []

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=10)
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
        kf = KFold(n_splits=10)
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


def knn_multi_output(x, y):

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    from sklearn.multioutput import MultiOutputClassifier
    clf = MultiOutputClassifier(KNeighborsClassifier(
        n_neighbors=1)).fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    for i in range(len(x_test)):
        print("Actual: ", true_false_to_genres(y_test[i]))
        print("Pred: ", true_false_to_genres(y_pred[i]))
        print()

    for i in range(len(BASE_GENRES)):
        auc = roc_auc_score(y_test[:, i], y_pred[:, i])
        print("AUC %s: %.4f" % (BASE_GENRES[i], auc))

    # cm_y1 = confusion_matrix(y_test[:, 0], y_pred[:, 0])
    # cm_y2 = confusion_matrix(y_test[:, 1], y_pred[:, 1])

    # print(cm_y1)
    # print(cm_y2)


def knn_cv_n_neighbours(x, y, neighbours):
    mean_error = []
    std_error = []

    for neighbour in neighbours:
        # kernel = create_gaussian_kernel(gamma)
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=neighbour)

        temp = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=10)
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
        kf = KFold(n_splits=10)
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
