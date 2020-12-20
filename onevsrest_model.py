import json
from util import BASE_GENRES, parse_data_multi_output, parse_data_single_output
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# WIP

# y = np.array([0, 0, 1, 1, 2, 2])
# clf = OneVsRestClassifier(SVC()).fit(X, y)
# clf.predict([[-19, -20], [9, 9], [-5, 5]])

with open('./data/2k_songs_sample_dataset.json') as f:
    data = json.load(f)
# for single output models
# x_train,y_train,x_hold_out,y_hold_out = parse_data_single_output(data)

# for multi output models
x_train, y_train, x_hold_out, y_hold_out = parse_data_single_output(data, True)
