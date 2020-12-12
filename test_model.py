import warnings
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import collections
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import BASE_GENRES
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcdefaults()


# Inspired from https: // towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
# first model example

# Read and format data from our dataset
df = pd.read_json('data/2k_songs_sample_dataset.json')

y = np.array([np.array(xi) for xi in df['genres']])
columns_to_use = df.columns.difference(['genres', 'type', 'id', 'uri',
                                        'track_href', 'analysis_url', 'artist', 'artist_id'])
X = np.array(df[columns_to_use])

# This is a HIGHLY inefficient way to map a set of genres into a set of boolean arrays per genre
# ie pop: [true, false, true...] - TODO : update this mess
mapped_y = []
for curr_song_genres in y:
    mapped_y.append({
        'pop': 'pop' in curr_song_genres,
        'rock': 'rock' in curr_song_genres,
        'rap': 'rap' in curr_song_genres,
        'dance': 'dance' in curr_song_genres,
        'hip': 'hip' in curr_song_genres,
        'trap': 'trap' in curr_song_genres,
        'r&b': 'r&b' in curr_song_genres,
        'metal': 'metal' in curr_song_genres,
        'country': 'country' in curr_song_genres,
        'indie': 'indie' in curr_song_genres,
        'folk': 'folk' in curr_song_genres,
        'alternative': 'alternative' in curr_song_genres,
        'punk': 'punk' in curr_song_genres,
        'electro': 'electro' in curr_song_genres,
        'psych': 'psych' in curr_song_genres,
    })
grouped_y = {
    'pop': [],
    'rock': [],
    'rap':  [],
    'dance':  [],
    'hip':  [],
    'trap':  [],
    'r&b':  [],
    'metal':  [],
    'country': [],
    'indie': [],
    'folk':  [],
    'alternative': [],
    'punk':  [],
    'electro':  [],
    'psych': [],
}
for item in mapped_y:
    for genre in item.keys():
        grouped_y[genre].append(item[genre])

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
])
for genre in BASE_GENRES:
    print('**Processing {} genre...**'.format(genre))
    # Training logistic regression model on train data
    LogReg_pipeline.fit(X, grouped_y[genre])

    # calculating test accuracy
    prediction = LogReg_pipeline.predict(X)
    print('Test accuracy is {}'.format(
        accuracy_score(grouped_y[genre], prediction)))
    print('\n')
