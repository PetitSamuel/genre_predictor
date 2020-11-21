import time

import spotipy
import json
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

scope = 'user-library-read'
# Should probably get your own client_id so we're not eating into each others rate limits
# look at https://developer.spotify.com/documentation/web-api/quick-start/
SPOTIPY_CLIENT_ID='3f137784ee774084b5f10cf1805e057b'
SPOTIPY_CLIENT_SECRET='61aefb1d48bc4391a11ee7e5c2d8b29b'


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
    track_ids = []        # in 100 track chunks
    artist_ids = []

    while True:
        response = sp.playlist_items(
            pl_uri,
            offset=offset,
            fields='items.track.id,items.track.artists.id,total',
            additional_types=['track'])
        pp_json(response)

        if len(response['items']) == 0:
            break
        for t in response['items']:
            t_id = t['track']['id']
            a_id = t['track']['artists'][0]['id']
            track_ids.append(t_id)
            artist_ids.append(a_id)

        offset = offset + len(response['items'])
        print(offset, "/", response['total'])

    return track_ids, artist_ids


# returns json of audio features given array of track ids
def get_audio_features(tracks):
    start = time.time()
    features = sp.audio_features(tracks)
    delta = time.time() - start
    print("features retrieved in %.2f seconds" % (delta))
    return features


# (unfinished)
def get_artist_genre(a_id):
    # offset = 0
    response = sp.artist(a_id)
    pp_json(response)


if __name__ == '__main__':
    # Sets up access keys basically
    # Should probably get your own client_id so we're not eating into each others rate limits
    # look at https://developer.spotify.com/documentation/web-api/quick-start/
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET)

    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # two test playlists
    small_pl_id = 'spotify:playlist:1v3dqlxEXvJHJ3YG94rxy9' # insp
    large_pl_id = 'spotify:playlist:5yzE4l99voJA0KMeomohzM' # coming of age
    tracks, artists = get_track_artist_ids_from_playlist(small_pl_id)
    print(tracks)
    print(artists)

    # get track features of max 100 tracks
    feat = get_audio_features(tracks[4])
    pp_json(feat[0])

    # get_artist_genre(artists[14])