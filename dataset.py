import collections
import pandas as pd
import os
import pycld2 as cld2

def fix_encoding(s):
    # Common character combinations in badly encoded latin-1
    latin_1_garbage = ["Ã¤","Ã©", "Ã¼", "Ã¬", "Ã±", "Ãª", "Ãº", "Î¼", 
        "â\x80", "Ç\x90", "Ã\x9c", "Å¾", "Ä\x8d", "Ã\x86", "Ã¹", "Ã²", "Ã¨",
        "Ã®", "Ã¢", "Å\x9f" ,"Ä\x83" ,"Å\x9f", "Å\x9b","Å\x82","Å\x9b", "Ã¥", "Ñ\x81", "Ã\x89"]
    for g in latin_1_garbage:
        if g in s:
            try:
                s = bytearray(s, 'latin-1').decode('utf-8')
            except:
                return None
            break
    # Remove / replace remaining unicode characters
    s = s.translate(str.maketrans('\u2028—\x84', '\n-"', "\u200b\x7f\x98\x9d\x90\x82\x18"))
    return s

def detect_language(s):
    try:
        _,_,res = cld2.detect(s)
        return res[0][1]
    except:
        return None

def load_genre_lyrics(genre, verbose=False):
    data_filepath = "lyrics.csv"

    if not os.path.exists(data_filepath):
        print("Missing dataset:",data_filepath)
        exit(1)

    lyrics_df = pd.read_csv(data_filepath)
    
    if (verbose): print(lyrics_df["genre"].unique())

    genre_df = lyrics_df[lyrics_df.genre == genre]

    if (verbose): print("Dropping", len(genre_df[genre_df["lyrics"].isnull()]), "missing lyrics")

    genre_df = genre_df.dropna(subset=["lyrics"])
    genre_df["len"] = genre_df["lyrics"].apply(len)

    if (verbose): print("Dropping", len(genre_df[genre_df["len"] < 1000]), "short lyrics")

    genre_df = genre_df[genre_df["len"] >= 1000]

    if (verbose): print("Fixing encodings")

    genre_df["lyrics"] = genre_df["lyrics"].apply(fix_encoding)

    if (verbose): print("Dropping", len(genre_df[genre_df["lyrics"].isnull()]), "unrecoverable")

    genre_df = genre_df.dropna(subset=["lyrics"])

    if (verbose): print("Detecting languages")

    genre_df["lang"] = genre_df["lyrics"].apply(detect_language)

    if (verbose): print("Could not detect on", len(genre_df[genre_df["lang"].isnull()]))
    if (verbose): print("Dropping", len(genre_df[genre_df["lang"] != "en"]), "not detected as English")

    genre_df = genre_df[genre_df["lang"] == "en"]
    genre_lyrics = genre_df.lyrics.values

    if (verbose): print("Songs left", len(genre_lyrics))
    if (verbose): print("Total size", sum(map(len, genre_lyrics)))

    return genre_lyrics