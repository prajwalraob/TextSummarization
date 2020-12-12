from tensorflow import keras


def main():
    fashion_dataset = keras.datasets.fashion_mnist
    (training_input_data, training_input_target), (testing_data, testing_target) = fashion_dataset.load_data()
    training_data = training_input_data[:5000] / 255.0
    validation_data = training_input_data[5000:] / 255.0
    training_target = training_input_target[:5000]
    validation_target = training_input_target[5000:]

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
    model.fit(training_data, training_target, epochs=20, batch_size=50, validation_data=(validation_data, validation_target))
    score = model.evaluate(testing_data, testing_target)

    print(score)


def main2():
    pass


if __name__ == '__main__':
    main()
