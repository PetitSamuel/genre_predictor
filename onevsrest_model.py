from sklearn.model_selection import train_test_split
import json
from util import BASE_GENRES, parse_data_multi_output, parse_data_single_output, genre_to_id
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.dummy import DummyClassifier


# get an array of 1 genre, map it to its number id
def map_to_genre_id(item):
    if len(item) == 0:
        return 0
    return genre_to_id(item[0])


with open('./data/11k_songs_tso_dataset.json') as f:
    data = json.load(f)

# for single output models
x, y, x_hold_out, y_hold_out = parse_data_single_output(data)
# Map the array of genres to a set of genre IDs
y = list(map(map_to_genre_id, y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = OneVsRestClassifier(SVC()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('SVC: ', accuracy)

clf = OneVsRestClassifier(RidgeClassifier()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Ridge: ', accuracy)

clf = OneVsRestClassifier(LogisticRegression()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Logistic: ', accuracy)


clf = OneVsRestClassifier(KNeighborsClassifier()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('kNN: ', accuracy)


clf = OneVsRestClassifier(DecisionTreeClassifier()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Decision Tree: ', accuracy)

clf = OneVsRestClassifier(DummyClassifier(
    strategy="most_frequent")).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Baseline: ', accuracy)

# param cross valid
# baselne model
# train / test
