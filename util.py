import json
import pandas as pd
import numpy as np
from sklearn import preprocessing

BASE_GENRES = [
    "pop",
    "rock",
    "rap",
    "dance",
    "hip",  # hip hop (catches "hip pop")
    "trap",
    "r&b",
    "metal",
    "country",
    "indie",
    "folk",
    "alternative",
    "punk",
    "electro",  # catches 'electronica', 'electro house',...
    "psych"
]


def genre_to_id(genre):
    return BASE_GENRES.index(genre) + 1


def show_tracks(results):
    for i, item in enumerate(results['items']):
        track = item['track']
        print(
            "   %d %32.32s %s" %
            (i, track['artists'][0]['name'], track['name']))


# pretty print json
def pp_json(json_thing, sort=True, indents=4):
    if type(json_thing) is str:
        print(json.dumps(json.loads(json_thing), sort_keys=sort, indent=indents))
    else:
        print(json.dumps(json_thing, sort_keys=sort, indent=indents))
    return None


# max = max labels per datum
def parse_data_multi_output(data, max):
    x, y = parse_data_multi_output_full(data, max)
    x_train = x[:10000]
    y_train = y[:10000]
    x_hold_out = x[10000:]
    y_hold_out = y[10000:]

    return x_train, y_train, x_hold_out, y_hold_out


def parse_data_multi_output_full(data, max):
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

    x = preprocessing.scale(np.array(x))
    y = map_true_false(y)
    y = np.array(y)

    from sklearn.utils import shuffle
    x, y = shuffle(x, y)

    return x, y


def parse_data_single_output(data, mapToBool=False):
    x, y = parse_data_single_output_full(data, mapToBool)
    x_train = x[:10000]
    y_train = y[:10000]
    x_hold_out = x[10000:]
    y_hold_out = y[10000:]

    return x_train, y_train, x_hold_out, y_hold_out


def parse_data_single_output_full(data, mapToBool=False):
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
            # print("skipping electro")
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

    if mapToBool:
        y = map_true_false(y)
    y = np.array(y)
    x = preprocessing.scale(np.array(x))

    from sklearn.utils import shuffle
    x, y = shuffle(x, y)

    return x, y


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
