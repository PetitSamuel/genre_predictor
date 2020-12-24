# Genre Predictor

## Set Up

### Spotify API Access

1. Get a Spotify API key at https://developer.spotify.com/documentation/web-api/quick-start/
2. Create a file called `.env` and add the following values:
    ```
    SPOTIFY_CLIENT_ID=<your_client_id>
    SPOTIFY_CLIENT_SECRET=<your_client_secret>
    ```


### Dependency Installation

```
pip3 install -r requirements.txt 
```

## Project Information
We were wondering how feasible it would be to predict the genre of a song using audio feature analysis metrics from the spotify API.
(https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/). These features include metrics such as: danceability, instrumentalness, acousticness...


### Fetching data

Running:
```
python3 data_fetcher.py
```

Will fetch the list of songs from an array of playlists defined around line 180. It will then proceed to fetch the audio features for all of these songs. Finally, it will fetch all of the artists related to these songs (such as to obtain the genres for each artist).

Spotify has thousands of genres, so we map those to a set of simplified genres while keeping order of their relevance.

The final dataset is dumped as a json file in the data directory.


### Data visualisation

Simple plots can be seen using:
```
python3 genre_plots.py # Count the appearance of each genre
python3 density_plots.py # Display the PDF for each of the audio metrics
```


### Machine Learning Models

#### OneVsRest approach

```
python3 onevsrest_model_single_output.py
```
Attempts to predict a single genre per song (the most relevant).

With multi output:
```
python3 onevsrest_model_multi_output.py
```
Trains a set of classifiers for each genre, automatically picks the model (and its associated hyperparameters) with the best accuracy,
then we use the union of all of the classifiers as our multi-label output.


#### Powerset Model

Predicts genres using the powerset (set of all subsets) of all genres, one classifier is trained per subset.

```
python3 label_powerset_model.py
```


#### kNN Multi output and Chain Classifier

Train a multi output kNN Model to predict multiple genres per song using the kNN model.
Also contains code for the Chain Classifier approach which will train a single classifier per genre, taking correlation into account by feeding the previously trained models into the one that is currently training.

```
python3 prediction_model.py
```

