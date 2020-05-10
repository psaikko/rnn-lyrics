import pandas as pd
import numpy as np
import random as rand
import os

lyrics_df = pd.read_csv("lyrics.csv")

print(lyrics_df.columns)
print(lyrics_df["genre"].unique())
print(len(lyrics_df))

GENRE = "Hip-Hop"

genre_df = lyrics_df[lyrics_df.genre == GENRE]
genre_df = genre_df.dropna(subset=["lyrics"])

genre_lyrics = genre_df.lyrics.values

print("Songs", len(genre_lyrics))
print("Characters", sum(map(len, genre_lyrics)))

alpha = list(chr(i) for i in range(ord('a'),ord('z')+1))
nums = list(str(i) for i in range(0,10));
punct = list(" ,.!?()\n")
START_SYMBOL = "^"
END_SYMBOL = "$"
allowed_chars = alpha + nums + punct

char_to_id = {c:i for (i,c) in enumerate(allowed_chars + [START_SYMBOL,END_SYMBOL])}
id_to_char = {i:c for (i,c) in enumerate(allowed_chars + [START_SYMBOL,END_SYMBOL])}

START_ID = char_to_id[START_SYMBOL]
END_ID = char_to_id[END_SYMBOL]

def preprocess(lyric):
    return [char_to_id[c] for c in lyric.lower() if c in allowed_chars]

def postprocess(indices):
    return ''.join(id_to_char[i] for i in indices)

M = 3
n_chars = len(allowed_chars)

filename = f"{GENRE}-{M}.npy"

if os.path.exists(filename):
    occ_mat = np.load(open(filename, "rb"))
else:
    occ_mat = np.zeros(shape=(n_chars+2,)*M, dtype=np.int32)
    for s in genre_lyrics:
        ixs = (START_ID,) * M
        for i in preprocess(s):
            ixs = ixs[1:] + (i,)
            occ_mat[ixs] += 1
        ixs = ixs[1:] + (END_ID,)
        occ_mat[ixs] += 1
    np.save(open(filename, "wb"), occ_mat)

print(np.sum(occ_mat))

ixs = (START_ID,) * (M-1)
while True:
    r = rand.random()
    row = occ_mat[ixs]
    s = np.sum(row)
    pos = s * r

    next_i = 0
    cum_sum = row[next_i]
    while cum_sum < pos:
        next_i += 1
        cum_sum += row[next_i] 

    ixs = ixs[1:] + (next_i,)
    if next_i == END_ID: break
    print(id_to_char[next_i],end='')
print()