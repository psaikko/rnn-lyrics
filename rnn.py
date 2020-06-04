import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import tensorflow as tf
import os
from tensorflow.keras import layers
import tensorflowjs as tfjs
import dataset

GENRE = "Country"
genre_lyrics = dataset.load_genre_lyrics(GENRE)

# https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-595467618
gpu_devices = tf.config.experimental.list_physical_devices('GPU') 
for device in gpu_devices: 
    tf.config.experimental.set_memory_growth(device, True)

data_size = 500
batch_size = 256
num_classes = 256
timesteps = 100
n_cells = 64
epochs = 2
rnn_layers = 2
celltype = 'SIMPLE'

print("Creating training slices")
X = []
y = []
# predicting (timesteps+1):th character from characters at 1..timesteps
for song in genre_lyrics[:data_size]:
    for i in range(len(song) - timesteps):
        X += [song[i:i+timesteps]]
        y += [song[i+timesteps]]

# char -> int
to_ord = lambda s: list(map(ord, s))
X = np.stack([np.array(x) for x in map(to_ord, X)], axis=0)
y = np.array(to_ord(y))

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, shuffle=True)

print(X_train.shape)
print(y_train.shape)

print("Data created")

def onehot_embedding_layer(n_classes, batch_input_shape):
    lookup_mat = tf.keras.utils.to_categorical(range(n_classes), n_classes)
    onehot_embed_weights = tf.keras.initializers.Constant(lookup_mat)
    return layers.Embedding(n_classes, n_classes, embeddings_initializer=onehot_embed_weights, trainable=False, batch_input_shape=batch_input_shape)

def make_model(stateful=False):
    print("Creating model")
    batch_input_shape = (1, 1) if stateful else (None, None)
    model = tf.keras.Sequential()
    model.add(onehot_embedding_layer(num_classes, batch_input_shape))
    for i in range(rnn_layers):
        last_rnn_layer = (i == rnn_layers - 1)
        model.add({
            'LSTM': layers.LSTM,
            'GRU': layers.GRU,
            'SIMPLE': layers.SimpleRNN
        }[celltype](n_cells, return_sequences=not last_rnn_layer, stateful=stateful))
    model.add(layers.Dense(num_classes, activation='linear'))

    loss = lambda labels, logits: \
        tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])
    print("Model created")
    return model

weights_filename = f"rnn-{GENRE}-{celltype}-{n_cells}-{rnn_layers}-{timesteps}-{data_size}x{epochs}.npy"

predict_model = make_model(stateful=True)

# load or compute model weight with current parameters
if os.path.exists(weights_filename):
    wts = np.load(open(weights_filename, "rb"), allow_pickle=True)
    predict_model.set_weights(wts)
else:
    train_model = make_model(stateful=False)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=1, verbose=1, min_lr=0.00001)
    history = train_model.fit(X_train, y_train, 
                            batch_size=batch_size,
                            epochs=epochs, 
                            shuffle=True,
                            steps_per_epoch=len(y_train)//batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[reduce_lr])
    # save for web
    tfjs.converters.save_keras_model(train_model, 'web/rnn_weights')
    trained_weights = train_model.get_weights()
    np.save(open(weights_filename, "wb"), trained_weights)
    predict_model.set_weights(trained_weights)

def encode(s):
    return to_ord(s)

# https://www.tensorflow.org/tutorials/text/text_generation
def sample_from(s, temperature=1):
    s = np.divide(s, temperature)
    return chr(tf.random.categorical(s, num_samples=1)[-1,0].numpy())

# testing different "temperatures" for sampling 
for temp in np.geomspace(0.001, 1, 10):
    text = "Who"
    predict_model.reset_states()
    for i in range(200):
        logits = predict_model.predict(np.expand_dims(encode(text[-1]), axis=0))
        c = sample_from(logits, temperature=temp)
        text += c
    print("Temp %.3f:" % temp, text)
