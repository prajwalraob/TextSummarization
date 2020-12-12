# https://towardsdatascience.com/text-summarization-from-scratch-using-encoder-decoder-network-with-attention-in-keras-5fa80d12710e

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
# import plaidml.keras
# import os
#
# plaidml.keras.install_backend()
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# import keras


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)


def test_single_layer_ann():
    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    x_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    x_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    x_test, y_test = series[9000:, :n_steps], series[9000:, -1]

    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=[50, 1]),
            keras.layers.Dense(1)
        ]
    )

    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test)
    print(score)


def main():
    n_steps = 50
    series = generate_time_series(10000, n_steps + 10)
    x_train, y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
    x_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
    x_test, y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

    model = keras.models.Sequential(
        [
            keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
            keras.layers.SimpleRNN(20),
            keras.layers.Dense(10)
        ]
    )

    model.compile(loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test)
    print(score)


def shakespeare():
    url = 'https://homl.info/shakespeare'
    # file_path = keras.utils.get_file('shakespeare.txt', url)
    file_path = "Napoleon.txt"
    text = ''
    with open(file_path) as f:
        text = f.read()
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts([text])
    [encode] = np.array(tokenizer.texts_to_sequences([text])) - 1
    dataset_size = encode.__len__()
    training_size = dataset_size * 90 // 100
    dataset = tf.data.Dataset.from_tensor_slices(encode[:training_size])

    print('Done')


def trials():
    pass


if __name__ == '__main__':
    trials()
