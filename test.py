import json
from util import pp_json

# Opening JSON file
f = open('data/2k_songs_sample_dataset.json',)

# Load from json into python array
data = json.load(f)
f.close()

# Print a random song
pp_json(data[1000])
