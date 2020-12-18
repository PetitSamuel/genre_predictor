import matplotlib.pyplot as plt
plt.rcdefaults()
import json
import collections

if __name__ == '__main__':
    # with open("data/2k_songs_sample_dataset.json") as f:
    with open("data/11k_songs_tso_dataset.json") as f:
        data = json.load(f)

    all_artist_genres = []
    for i in data:
        list_genres = i['genres'][:2]
        for g in list_genres:   # max first 2 genres of an artist
            all_artist_genres.append(g)

    counts = collections.Counter(all_artist_genres)
    x = list(counts.keys())
    y = list(counts.values())

    # sort list so it looks nice
    unsorted_list = [(y, x) for x, y in
                     zip(x, y)]
    sorted_list = sorted(unsorted_list, reverse=True)

    x = []
    y = []
    for i in sorted_list:
        x.append(i[1])
        y.append(i[0])

    plt.barh(x, y, align='center', alpha=0.5)
    plt.show()
