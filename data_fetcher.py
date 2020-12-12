import time
import spotipy
import json
from more_itertools import unique_everseen
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from decouple import config
from util import pp_json, BASE_GENRES

SPOTIFY_CLIENT_ID = config('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = config('SPOTIFY_CLIENT_SECRET')

scope = 'user-library-read'

# artist_genre_map = {
#     "5INjqkS1o8h1imAzPqGZBb": {
#         "artist": "Tame Impala",
#         "genres": ["psych"],
#         "artist_id": <5INjqkS1o8h1imAzPqGZBb>
#     }
# }
artist_genre_map = {}
# track_to_artist_map = {
#   "<track id>": <artist id>
# }
track_to_artist_map = {}


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
            if t['track'] is not None:
                t_id = t['track']['id']
                a_id = t['track']['artists'][0]['id']
                if t_id and a_id:
                    track_ids.append(t_id)
                    artist_ids.append(a_id)
                    track_to_artist_map[t_id] = a_id
        offset = offset + len(response['items'])
        # print(offset, "/", response['total'])

    return track_ids, artist_ids


# returns json of audio features given array of track ids
def get_audio_features(tracks):
    start = time.time()
    features = sp.audio_features(tracks)
    delta = time.time() - start
    # print("features retrieved in %.2f seconds" % (delta))
    return features


# Foals
# ['alternative dance', 'indie rock', 'modern alternative rock', 'modern rock', 'new rave', 'oxford indie', 'rock']
# ['alternative'x2, 'dance'x1, 'indie'x2, 'rock'x4, 'modern'x2, 'new'x1, 'rave'x1, 'oxford'x1]
# ['rock', 'alternative', 'indie', 'modern', 'dance', 'new', 'rave', 'oxford']

# Change so we keep track of how many "rock"s we see and order based on that
# So Foals should be
# ['rock', 'indie', ...]
def tokenize_genres(genres):
    genre_tokens = []
    for g in genres:
        if "-" in g:
            hyphen_split = g.split("-")
            g = tokenize_genres(hyphen_split)
            for subg in g:
                # if subg not in genre_tokens:
                genre_tokens.append(subg)
        else:
            space_split = g.split()
            for subg in space_split:
                # if subg not in genre_tokens:
                genre_tokens.append(subg)

    if genre_tokens == []:
        return []

    return order_genre_list(genre_tokens)


# Takes a full list of genre subtokens
# Orders by frequency and returns unique list
def order_genre_list(lst):
    from collections import Counter
    counts = Counter(lst)
    if max(counts.values()) != 1:
        lst = sorted(lst, key=lambda x: (counts[x], x), reverse=True)
    lst = list(unique_everseen(lst))

    return lst


# parses spotify's genres and returns genres from BASE_GENRES
def analyse_genres(genres):
    genre_tokens = tokenize_genres(genres)

    artist_base_genres = []
    for genre in genre_tokens:
        if genre in BASE_GENRES:
            artist_base_genres.append(genre)

    return artist_base_genres


# Get a list of artist ids, fetch the ones currently not in the artist genre map and add them to the map.
def process_artist_batch(artists):
    # Only keep artists which are not currently in the genre map
    artists_to_fetch = [
        a_id for a_id in artists if a_id not in artist_genre_map]

    # fetch remaining artists
    if len(artists_to_fetch) == 0:
        return
    response = sp.artists(artists_to_fetch)

    # Add fetched artists to artist genre map
    if len(response['artists']) == 0:
        return
    for curr_artist in response['artists']:
        spotify_genres = curr_artist['genres']
        name = curr_artist['name']
        a_id = curr_artist['id']
        parsed_genres = analyse_genres(spotify_genres)
        artist_genre_map[a_id] = {
            "artist": name,
            "genres": parsed_genres,
            # Add artist ID here for easier merging of data
            "artist_id": a_id
        }


# Take all audio features, retrieve the artist from the track to artist map
# and merge both objects into a single one. return the list of finalised data points.
def merge_tracks_genres(all_audio_features):
    all_data = []
    if all_audio_features is None or len(all_audio_features) == 0:
        return []
    for track in all_audio_features:
        artist_id = track_to_artist_map[track['id']]
        artist = artist_genre_map[artist_id]
        if artist is None:
            print("Error: couldn't find artist id: " + artist_id)
            continue
        merged = dict()
        merged.update(track)
        merged.update(artist)
        all_data.append(merged)
    return all_data


# Take an array and split it in batches of n (default size is 100)
def make_batch(iterable, n=100):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:min(i + n, l)]


if __name__ == '__main__':
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET)

    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Set of playlists
    small_pl_id = 'spotify:playlist:1v3dqlxEXvJHJ3YG94rxy9'  # insp
    large_pl_id = 'spotify:playlist:5yzE4l99voJA0KMeomohzM'  # coming of age
    irish_top_50 = 'spotify:playlist:37i9dQZEVXbKM896FDX8L1'
    folk = 'spotify:playlist:37i9dQZF1DX6z20IXmBjWI'
    large_pl_id_2 = 'spotify:playlist:4lJjt8XIq6zYphoMjNJIb4'

    # from everynoise.com ('tso' = the sound of...)
    tso_pop = 'spotify:playlist:6gS3HhOiI17QNojjPuPzqc'
    tso_rock = 'spotify:playlist:7dowgSWOmvdpwNkGFMUs6e'
    tso_rap = 'spotify:playlist:6s5MoZzR70Qef7x4bVxDO1'
    tso_dance_pop = 'spotify:playlist:2ZIRxkFuqNPMnlY7vL54uK' #nothing for just dance
    tso_hiphop = 'spotify:playlist:6MXkE0uYF4XwU4VTtyrpfP'
    tso_trap = 'spotify:playlist:60SHtDyagDjPnUpC7x1UD9'
    tso_rnb = 'spotify:playlist:1rLnwJimWCmjp3f0mEbnkY'
    tso_metal = 'spotify:playlist:3pBfUFu8MkyiCYyZe849Ks'
    tso_country = 'spotify:playlist:0VZfpqcbBUWC6kpP1vVrvA'
    tso_indietronica = 'spotify:playlist:0yqVOsxA2U4P260ad60QuU'
    tso_folk = 'spotify:playlist:4JuKjgd76AZn2fUaeXNCuo'
    # alternative + whatever we're missing
    tso_alternative_dance = 'spotify:playlist:5LwcdWTCx2JoWeVVWOYsGj'
    tso_punk = 'spotify:playlist:17qQT0G3yFjOJ02wWZaNCw'
    tso_electronica = 'spotify:playlist:6I0NsYzfoj7yHXyvkZYoRx'
    tso_chamber_psych = 'spotify:playlist:6rirvdbul7rDunT5SP5F4m'

    all_tso = [
        tso_pop,tso_rock,tso_rap,tso_dance_pop,tso_hiphop,tso_trap,tso_rnb,tso_metal,
        tso_country,tso_indietronica,tso_folk,tso_punk,tso_electronica,tso_chamber_psych,
        tso_alternative_dance
    ]

    # Array of playlists IDs to fetch
    all_playlists = [
        small_pl_id, large_pl_id, large_pl_id_2, irish_top_50, folk
    ]
    # Use this for small tests
    single_playlist = [small_pl_id]

    print("Fetching tracks from playlists...")

    # Gets artists and tracks IDs
    all_tracks = []
    all_artists = []
    for playlist_id in all_tso:
        tracks, artists = get_track_artist_ids_from_playlist(playlist_id)
        all_tracks = all_tracks + tracks
        all_artists = all_artists + artists

    print("Found " + str(len(all_tracks)) + " tracks")
    print("Fetching audio features for all tracks...")

    # get track features for all tracks
    all_audio_features = []
    for tracks in make_batch(all_tracks):
        all_audio_features = all_audio_features + get_audio_features(tracks)

    print("Fetching artists...")

    # get artists genres
    for artists in make_batch(all_artists, 50):
        process_artist_batch(artists)

    print("Merging data...")

    # Merge both track features and genres
    merged_data = merge_tracks_genres(all_audio_features)

    print("Writing output to file...")

    # write to file, use current time to generate a unique file
    epoch_time = int(time.time())
    minified_json = json.dumps(merged_data, separators=(',', ':'))
    f = open("data/dataset_" + str(epoch_time) + ".json", "w")
    f.write(minified_json)
    f.close()

    print("Finished!")
