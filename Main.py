import time
import spotipy
import json
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from decouple import config

SPOTIFY_CLIENT_ID = config('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = config('SPOTIFY_CLIENT_SECRET')

scope = 'user-library-read'

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


def get_track_artist_ids_from_playlist(pl_uri):
    offset = 0
    track_ids = []  # in 100 track chunks
    artist_ids = []

    while True:
        response = sp.playlist_items(
            pl_uri,
            offset=offset,
            fields='items.track.id,items.track.artists.id,total',
            additional_types=['track'])
        # pp_json(response)

        if len(response['items']) == 0:
            break
        for t in response['items']:
            t_id = t['track']['id']
            a_id = t['track']['artists'][0]['id']
            track_ids.append(t_id)
            artist_ids.append(a_id)

        offset = offset + len(response['items'])
        # print(offset, "/", response['total'])

    return track_ids, artist_ids


# returns json of audio features given array of track ids
def get_audio_features(tracks):
    start = time.time()
    features = sp.audio_features(tracks)
    delta = time.time() - start
    print("features retrieved in %.2f seconds" % (delta))
    return features


BASE_GENRES = [
    "pop"
    , "rock"
    , "rap"
    , "dance"
    , "hip"  # hip hop (catches "hip pop")
    , "trap"
    , "r&b"
    , "metal"
    , "country"
    , "indie"
    , "folk"
    , "alternative"
    , "punk"
    , "electro"  # catches 'electronica', 'electro house',...
    , "psych"
]

# artist_genre_map = {
#     "5INjqkS1o8h1imAzPqGZBb": {
#         "name": "Tame Impala",
#         "genres": ["psych"]
#     }
# }

artist_genre_map = {}

# example Foals genres are
# ['alternative dance', 'indie rock', 'modern alternative rock', 'modern rock', 'new rave', 'oxford indie', 'rock']
# Returns unique subtokens
# ['alternative', 'dance', 'indie', 'rock', 'modern', 'new', 'rave', 'oxford']
def tokenize_genres(genres):
    genre_tokens = []
    for g in genres:
        if "-" in g:
            hyphen_split = g.split("-")
            g = tokenize_genres(hyphen_split)
            for subg in g:
                if subg not in genre_tokens:
                    genre_tokens.append(subg)
        else:
            space_split = g.split()
            for subg in space_split:
                if subg not in genre_tokens:
                    genre_tokens.append(subg)

    return genre_tokens


# parses spotify's genres and returns genres from BASE_GENRES
def analyse_genres(genres):
    genre_tokens = tokenize_genres(genres)

    artist_base_genres = []
    for bg in BASE_GENRES:
        if bg in genre_tokens:
            artist_base_genres.append(bg)

    return artist_base_genres

# Check if artist and genres in the dict
# If not, makes a call and processes them
# adds to the dict
# Returns a list of base genres the artist is a part of
def get_artist_genres(a_id):
    # Check if the artist and its genres in dict already
    if a_id in artist_genre_map:
        print("hit")
        return artist_genre_map[a_id]

    # If not then fetch artist and parse genres
    response = sp.artist(a_id)

    spotify_genres = response['genres']
    name = response['name']

    parsed_genres = analyse_genres(spotify_genres)
    artist_genre_map[a_id] = {
        "name":name,
        "genres":parsed_genres
    }

    return parsed_genres


if __name__ == '__main__':
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET)

    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # two test playlists
    small_pl_id = 'spotify:playlist:1v3dqlxEXvJHJ3YG94rxy9'  # insp
    large_pl_id = 'spotify:playlist:5yzE4l99voJA0KMeomohzM'  # coming of age
    irish_top_50 = 'spotify:playlist:37i9dQZEVXbKM896FDX8L1'
    folk = 'spotify:playlist:37i9dQZF1DX6z20IXmBjWI'

    # Run through of one track

    # Gets artist and tracks IDs

    tracks, artists = get_track_artist_ids_from_playlist(small_pl_id)

    # get track features of max 100 tracks
    feat = get_audio_features(tracks[7])

    # get genres of artist
    genres = get_artist_genres(artists[7])

    data = {"features": feat[0], "genres": genres}
    json_data = json.dumps(data)

    pp_json(json_data)

