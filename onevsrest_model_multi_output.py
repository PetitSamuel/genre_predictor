from sklearn.model_selection import train_test_split
import json
from util import BASE_GENRES, parse_data_multi_output, parse_data_single_output, genre_to_id, get_precision, get_f1, get_tpr
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, classification_report
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
plt.rcdefaults()


with open('./data/11k_songs_tso_dataset.json') as f:
    data = json.load(f)

x, y, x_hold_out, y_hold_out = parse_data_multi_output(data, len(BASE_GENRES))
# Split data into train / test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

best_models = []
best_models_accuracy = []
best_models_conf_mat = []
# Train models for each possible genre
for i in range(len(BASE_GENRES)):
    print("CURRENT GENRE: ", BASE_GENRES[i])
    # The best predictor out of all models for the current genre
    best_predictor = None
    # Plot bar chart for the best accuracy vs model type
    bar_chart_labels = []
    bar_chart_scores = []

    y_genre_train = y_train[:, i]
    y_genre_test = y_test[:, i]
    # SVC model
    C_range = [0.0001, 0.01, 0.1, 1, 5, 10, 20, 30, 40, 50, 75, 100]
    scores = []
    for C in C_range:
        clf = SVC(C=C).fit(x_train, y_genre_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_genre_test, y_pred)
        scores.append(accuracy)
        if best_predictor == None or accuracy > best_predictor[1]:
            conf_mat = confusion_matrix(y_genre_test, y_pred)
            best_predictor = (clf, accuracy, conf_mat)
        print('SVC: ', accuracy, '  C: ', C)

    bar_chart_labels.append('SVC')
    bar_chart_scores.append(max(scores))

    # plot the CV
    fig = plt.figure()
    plt.rc('font', size=20)
    ax = fig.add_subplot(111)
    ax.errorbar(C_range, scores, label='Mean accuracy')
    ax.set_ylabel("Score (mean accuracy) of the model")
    ax.set_xlabel("C")
    ax.set_title(
        "C Cross-validation | SVC model | " + BASE_GENRES[i])
    ax.legend(loc='lower right')
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=18)

    alpha_range = [0.0001, 0.01, 0.1, 0, 1, 5, 10, 20, 30, 40, 50, 75, 100]
    scores = []
    for alpha in alpha_range:
        clf = RidgeClassifier(alpha=alpha).fit(x_train, y_genre_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_genre_test, y_pred)
        scores.append(accuracy)
        print('Ridge: ', accuracy, '  C: ', C)
        if best_predictor == None or accuracy > best_predictor[1]:
            conf_mat = confusion_matrix(y_genre_test, y_pred)
            best_predictor = (clf, accuracy, conf_mat)

    bar_chart_labels.append('Ridge')
    bar_chart_scores.append(max(scores))

    # plot the CV
    fig = plt.figure()
    plt.rc('font', size=20)
    ax = fig.add_subplot(111)
    ax.errorbar(alpha_range, scores, label='Mean accuracy')
    ax.set_ylabel("Score (mean accuracy) of the model")
    ax.set_xlabel("Alpha")
    ax.set_title(
        "Alpha Cross-validation | Ridge model | " + BASE_GENRES[i])
    ax.legend(loc='lower right')
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=18)

    C_range = [0.0001, 0.01, 0.1, 1, 5, 10, 20, 30, 40, 50, 75, 100]
    scores = []
    mses = []
    for C in C_range:
        clf = LogisticRegression(C=C).fit(x_train, y_genre_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_genre_test, y_pred)
        scores.append(accuracy)
        print('Logistic: ', accuracy, '  C: ', C)
        if best_predictor == None or accuracy > best_predictor[1]:
            conf_mat = confusion_matrix(y_genre_test, y_pred)
            best_predictor = (clf, accuracy, conf_mat)

    bar_chart_labels.append('Logistic')
    bar_chart_scores.append(max(scores))

    # plot the CV
    fig = plt.figure()
    plt.rc('font', size=20)
    ax = fig.add_subplot(111)
    ax.errorbar(C_range, scores, label='Mean accuracy')
    ax.set_ylabel("Score (mean accuracy) of the model")
    ax.set_xlabel("C")
    ax.set_title(
        "C Cross-validation | Logistic model | " + BASE_GENRES[i])
    ax.legend(loc='lower right')
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=18)

    neighbors_range = [1, 2, 4, 5, 8, 10, 15, 20, 30, 50]
    scores = []
    for n in neighbors_range:
        clf = KNeighborsClassifier(
            n_neighbors=n).fit(x_train, y_genre_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_genre_test, y_pred)
        scores.append(accuracy)
        print('kNN: ', accuracy, '  n: ', n)
        if best_predictor == None or accuracy > best_predictor[1]:
            conf_mat = confusion_matrix(y_genre_test, y_pred)
            best_predictor = (clf, accuracy, conf_mat)

    bar_chart_labels.append('kNN')
    bar_chart_scores.append(max(scores))

    # plot the CV
    fig = plt.figure()
    plt.rc('font', size=20)
    ax = fig.add_subplot(111)
    ax.errorbar(neighbors_range, scores, label='Mean accuracy')
    ax.set_ylabel("Score (mean accuracy) of the model")
    ax.set_xlabel("n")
    ax.set_title(
        "n Neighbors Cross-validation | kNN model | " + BASE_GENRES[i])
    ax.legend(loc='lower right')
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=18)

    clf = DecisionTreeClassifier().fit(x_train, y_genre_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_genre_test, y_pred)
    print('Decision Tree: ', accuracy)
    if best_predictor == None or accuracy > best_predictor[1]:
        conf_mat = confusion_matrix(y_genre_test, y_pred)
        best_predictor = (clf, accuracy, conf_mat)

    bar_chart_labels.append('Decision Tree')
    bar_chart_scores.append(accuracy)

    clf = DummyClassifier(
        strategy="most_frequent").fit(x_train, y_genre_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_genre_test, y_pred)
    print('Baseline (most frequent): ', accuracy)
    if best_predictor == None or accuracy > best_predictor[1]:
        conf_mat = confusion_matrix(y_genre_test, y_pred)
        best_predictor = (clf, accuracy, conf_mat)

    bar_chart_labels.append('Baseline')
    bar_chart_scores.append(accuracy)

    # Plot the bar chart for each model type
    bar_x_pos = [i for i, _ in enumerate(bar_chart_labels)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(bar_chart_labels, bar_chart_scores, color='green')
    ax.set_xlabel("Model")
    ax.set_ylabel("Best Score (accuracy)")
    ax.set_title("Best score for each model for genre " + BASE_GENRES[i])

    best_models.append(best_predictor[0])
    best_models_accuracy.append(best_predictor[1])
    best_models_conf_mat.append(best_predictor[2])

print("best models in order of genres: ", best_models)
print("best models accuracy in same order: ", best_models_accuracy)
print("best models confusion matrix in same order: ", best_models_conf_mat)

# Make a union of all classifiers outputs for each songs
TP = 0
TN = 0
FP = 0
FN = 0
for track_index in range(len(x_test)):
    x_track = x_test[track_index]
    y_track = y_test[track_index]
    # For each track, for each genre, count the right and wrong predictions
    for genre_index in range(len(BASE_GENRES)):
        pred = best_models[genre_index].predict([x_track])
        if y_track[genre_index] == True and pred[0] == y_track[genre_index]:
            TP += 1
        elif y_track[genre_index] == False and pred[0] == y_track[genre_index]:
            TN += 1
        elif pred[0] == True:
            FP += 1
        else:
            FN += 1

print("\nConfusion Matrix")
print(TP, TN)
print(FP, FN)
print("\nPrecision and F1 metric: ")
print(get_precision(TP, FP), get_f1(TP, FP, FN))
plt.show()
