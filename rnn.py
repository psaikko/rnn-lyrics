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

data_size = 200
batch_size = 32
num_classes = 256
timesteps = 10
units = 128

X = []
y = []
# predicting (timesteps+1):th character from characters at 1..timesteps
for song in genre_lyrics[:data_size]:
    for i in range(len(song) - timesteps):
        X += [song[i:i+timesteps]]
        y += [song[i+timesteps]]

# char -> int
to_ord = lambda s: list(map(ord, s))
X = list(map(to_ord, X))
y = to_ord(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

def onehot_embedding_layer(n_classes, batch_input_shape):
    lookup_mat = tf.keras.utils.to_categorical(range(n_classes), n_classes)
    onehot_embed_weights = tf.keras.initializers.Constant(lookup_mat)
    return layers.Embedding(n_classes, n_classes, embeddings_initializer=onehot_embed_weights, trainable=False, batch_input_shape=batch_input_shape)

def make_model(stateful=False):
    batch_input_shape = (batch_size, timesteps) if stateful else (None, None)

    model = tf.keras.Sequential()
    model.add(onehot_embedding_layer(num_classes, batch_input_shape))
    model.add(layers.SimpleRNN(units, return_sequences=False, stateful=stateful))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

weights_filename = f"rnn-{GENRE}-{timesteps}-{data_size}.npy"

predict_model = make_model(stateful=True)

# load or compute model weight with current parameters
if os.path.exists(weights_filename):
    wts = np.load(open(weights_filename, "rb"), allow_pickle=True)
    predict_model.set_weights(wts)
else:
    train_model = make_model(stateful=False)
    history = train_model.fit(X_train, y_train, 
                            batch_size=batch_size,
                            epochs=1, 
                            steps_per_epoch=len(y_train)//batch_size,
                            validation_data=(X_test, y_test)) 
    trained_weights = train_model.get_weights()
    np.save(open(weights_filename, "wb"), trained_weights)
    predict_model.set_weights(trained_weights)

def encode(s):
    return to_ord(s)

def sample_from(s, temperature=1):
    s = np.array(s)
    s = np.divide(s, temperature)
    s = scipy.special.softmax(s)
    # Sample from distribution s
    return chr(tf.random.categorical(tf.math.log(s), 1)[0][0].numpy())

# testing different "temperatures" for sampling
for temp in np.geomspace(0.001, 0.05, 10):
    text = "I"
    predict_model.reset_states()
    for i in range(100):
        pred = predict_model.predict(np.expand_dims(encode(text[-1]), axis=0))
        c = sample_from(pred, temperature=temp)
        text += c
    print("Temp %.3f: " % temp, text[-101:].replace("\n", "\\n"))
