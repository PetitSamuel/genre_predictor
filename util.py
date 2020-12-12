import json

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
