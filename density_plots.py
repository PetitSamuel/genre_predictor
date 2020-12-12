import collections
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()

# Read and format data from our dataset
df = pd.read_json("data/11k_songs_tso_dataset.json")

attributes = ['danceability', 'energy',
              'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

# Show all datapoints
for at in attributes:
    col = np.array(df[at])

    # Draw the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(col, bins=30, color='blue', edgecolor='black')

    # Title and labels
    ax.set_title(at + ' Histogram | 30 bins', size=30)
    ax.set_xlabel('test', size=22)
    ax.set_ylabel(at, size=22)

plt.show()
