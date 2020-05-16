import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy 
import sklearn.model_selection
import tensorflow as tf
import random
import os
from tensorflow.keras import layers

data_filepath = "lyrics.csv"
lyrics_df = pd.read_csv(data_filepath)
GENRE = "Hip-Hop"
print(lyrics_df["genre"].unique())
genre_df = lyrics_df[lyrics_df.genre == GENRE]
genre_df = genre_df.dropna(subset=["lyrics"])
genre_lyrics = genre_df.lyrics.values
print("Songs", len(genre_lyrics))
print("Characters", sum(map(len, genre_lyrics)))

batch_size = 128
num_classes = 256
timesteps = 10
units = 128

X = []
y = []
# predicting (timesteps+1):th character from characters at 1..timesteps
for song in genre_lyrics[:1000]:
    for i in range(len(song) - timesteps):
        X += [song[i:i+timesteps]]
        y += [song[i+timesteps]]

# char -> int
to_ord = lambda s: list(map(ord, s))
X = list(map(to_ord, X))
y = to_ord(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

def gen(xs, ys, batch_size):
    pair_gen = zip(xs, ys)
    while True:
        x_batch = np.zeros((batch_size, timesteps, num_classes)) 
        y_batch = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            x, y = next(pair_gen)
            y_batch[i,:] = tf.keras.utils.to_categorical([y], num_classes=num_classes)
            x_batch[i,:,:] = tf.keras.utils.to_categorical(x, num_classes=num_classes)

        yield (x_batch, y_batch)

model = tf.keras.Sequential()
model.add(layers.SimpleRNN(units, input_dim=num_classes, return_sequences=False, stateful=False))
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

weights_filename = f"rnn-{GENRE}-weights.npy"

# load or compute state-transition matrix from dataset
if os.path.exists(weights_filename):
    wts = np.load(open(weights_filename, "rb"), allow_pickle=True)
    model.set_weights(wts)
else:
    history = model.fit(x=gen(X_train, y_train, batch_size), 
                        epochs=1, 
                        steps_per_epoch=len(y_train)//batch_size)
    np.save(open(weights_filename, "wb"), model.get_weights())

    print(model.evaluate(x=gen(X_test, y_test, batch_size)))

def encode(s):
    return tf.keras.utils.to_categorical(to_ord(s), num_classes=num_classes)

def sample_from(s, temperature=1):
    s = np.array(s)
    s = np.divide(s, temperature)
    s = scipy.special.softmax(s)
    # Sample from distribution s
    return chr(tf.random.categorical(tf.math.log(s), 1)[0][0].numpy())

text = "I"
for i in range(10):
    for i in range(100):
        pred = model.predict(tf.convert_to_tensor(np.expand_dims(encode(text[-timesteps:]), axis=0)))
        c = sample_from(pred, temperature=0.02)
        text += c
    print(text[-100:])
