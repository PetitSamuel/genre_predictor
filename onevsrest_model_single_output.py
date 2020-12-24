from sklearn.model_selection import train_test_split
import json
from util import BASE_GENRES, parse_data_multi_output, parse_data_single_output, genre_to_id
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, precision_recall_fscore_support
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
plt.rcdefaults()


# get an array of 1 genre, map it to its number id
def map_to_genre_id(item):
    if len(item) == 0:
        return 0
    return genre_to_id(item[0])


with open('./data/11k_songs_tso_dataset.json') as f:
    data = json.load(f)

x, y, x_hold_out, y_hold_out = parse_data_single_output(data)
# Map the array of genres to a set of genre IDs
y = list(map(map_to_genre_id, y))
# Split dataset into train / test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# List of accuracy / model type for a bar chart comparison
bar_chart_labels = []
bar_chart_scores = []

# The model with the best accuracy
best_model = None

C_range = [0.0001, 0.01, 0.1, 1, 5, 10, 20, 30, 40, 50, 75, 100]
scores = []
mses = []
for C in C_range:
    clf = OneVsRestClassifier(SVC(C=C)).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    mses.append(mean_squared_error(y_test, y_pred))
    print('SVC: ', accuracy, '  C: ', C)
    if best_model == None or accuracy > best_model[1]:
        conf_mat = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        best_model = (clf, accuracy, conf_mat)

bar_chart_labels.append('SVC')
bar_chart_scores.append(max(scores))

# plot the CV
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(C_range, scores, label='Mean accuracy', yerr=mses)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("C")
ax.set_title(
    "C Cross-validation | SVC model")
ax.legend(loc='lower right')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)

alpha_range = [0.0001, 0.01, 0.1, 0, 1, 5, 10, 20, 30, 40, 50, 75, 100]
scores = []
mses = []
for alpha in alpha_range:
    clf = OneVsRestClassifier(RidgeClassifier(
        alpha=alpha)).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    mses.append(mean_squared_error(y_test, y_pred))
    print('Ridge: ', accuracy, '  C: ', C)
    if best_model == None or accuracy > best_model[1]:
        conf_mat = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        best_model = (clf, accuracy, conf_mat)

bar_chart_labels.append('Ridge')
bar_chart_scores.append(max(scores))

# plot the CV
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(alpha_range, scores, label='Mean accuracy', yerr=mses)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("Alpha")
ax.set_title(
    "Alpha Cross-validation | Ridge model")
ax.legend(loc='lower right')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)


C_range = [0.0001, 0.01, 0.1, 1, 5, 10, 20, 30, 40, 50, 75, 100]
scores = []
mses = []
for C in C_range:
    clf = OneVsRestClassifier(LogisticRegression(C=C)).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    mses.append(mean_squared_error(y_test, y_pred))
    print('Logistic: ', accuracy, '  C: ', C)
    if best_model == None or accuracy > best_model[1]:
        conf_mat = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        best_model = (clf, accuracy, conf_mat)

bar_chart_labels.append('Logistic')
bar_chart_scores.append(max(scores))

# plot the CV
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(C_range, scores, label='Mean accuracy', yerr=mses)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("C")
ax.set_title(
    "C Cross-validation | Logistic model")
ax.legend(loc='lower right')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)

neighbors_range = [1, 2, 4, 5, 8, 10, 15, 20, 30, 50]
scores = []
mses = []
for n in neighbors_range:
    clf = OneVsRestClassifier(KNeighborsClassifier(
        n_neighbors=n)).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    mses.append(mean_squared_error(y_test, y_pred))
    print('kNN: ', accuracy, '  n: ', n)
    if best_model == None or accuracy > best_model[1]:
        conf_mat = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        best_model = (clf, accuracy, conf_mat)

bar_chart_labels.append('kNN')
bar_chart_scores.append(max(scores))

# plot the CV
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(neighbors_range, scores, label='Mean accuracy', yerr=mses)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("n")
ax.set_title(
    "n Neighbors Cross-validation | kNN model")
ax.legend(loc='lower right')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)

clf = OneVsRestClassifier(DecisionTreeClassifier()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Decision Tree: ', accuracy)
if best_model == None or accuracy > best_model[1]:
    conf_mat = precision_recall_fscore_support(
        y_test, y_pred, average='weighted')
    best_model = (clf, accuracy, conf_mat)

bar_chart_labels.append('Decision Tree')
bar_chart_scores.append(accuracy)

clf = OneVsRestClassifier(DummyClassifier(
    strategy="most_frequent")).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Baseline (most frequent): ', accuracy)
if best_model == None or accuracy > best_model[1]:
    conf_mat = precision_recall_fscore_support(
        y_test, y_pred, average='weighted')
    best_model = (clf, accuracy, conf_mat)

bar_chart_labels.append('Baseline')
bar_chart_scores.append(accuracy)

bar_x_pos = [i for i, _ in enumerate(bar_chart_labels)]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(bar_chart_labels, bar_chart_scores, color='green')
ax.set_xlabel("Model")
ax.set_ylabel("Best Score (accuracy)")

# Print the model with the highest accuracy, its accuracy and precision metrics (precision recall, f1...)
print(best_model)
plt.show()
