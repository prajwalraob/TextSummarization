
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, GRUCell, RNN, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
import pandas


class SummarizationTrial:
    def __init__(self):
        self.text_data = object
        self.summary_data = object
        self.summary_data_target = object

        self.text_data_train = []
        self.text_data_valid = []
        self.text_data_test = []
        self.summary_data_train = []
        self.summary_data_valid = []
        self.summary_data_test = []
        self.summary_data_target_train = []
        self.summary_data_target_valid = []
        self.summary_data_target_test = []

        self.character_set_size = -1
        self.model_name = 'gru_10_1000.h5'

    def read_data(self):
        step1 = 900
        step2 = 1000
        step3 = 1050
        data_frame = pandas.read_csv('Reviews.csv')
        data = data_frame[['Text', 'Summary']][:step3]
        texts = [string for string in data.Text]
        max_text_length = max(len(string) for string in texts)
        summaries = [('\t' + string + '\t') for string in data.Summary]
        max_summary_length = max(len(string) for string in summaries)
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(texts)
        tokenizer.fit_on_texts(summaries)
        self.character_set_size = len(tokenizer.word_index)

        self.text_data = np.zeros((len(texts), max_text_length, self.character_set_size), dtype='float32')
        self.summary_data = np.zeros((len(summaries), max_summary_length, self.character_set_size), dtype='float32')
        self.summary_data_target = np.zeros((len(summaries), max_summary_length, self.character_set_size), dtype='float32')

        for i, value in enumerate(texts):
            for j, character in enumerate(value):
                self.text_data[i, j, tokenizer.word_index[character.lower()] - 1] = 1

        for i, value in enumerate(summaries):
            for j, character in enumerate(value):
                self.summary_data[i, j, tokenizer.word_index[character.lower()] - 1] = 1
                if j > 0:
                    self.summary_data_target[i, j-1, tokenizer.word_index[character.lower()] - 1] = 1

        self.text_data_train = self.text_data[:step2]
        self.text_data_valid = self.text_data[step1:step2]
        self.text_data_test = self.text_data[step2:step3]

        self.summary_data_train = self.summary_data[:step2]
        self.summary_data_valid = self.summary_data[step1:step2]
        self.summary_data_test = self.summary_data[step2:step3]

        self.summary_data_target_train = self.summary_data[:step2]
        self.summary_data_target_valid = self.summary_data[step1:step2]
        self.summary_data_target_test = self.summary_data[step2:step3]

    def create_encoder_decoder(self):
        num_cells = 256
        encoder_inputs = Input(shape=(None, self.character_set_size))
        encoder = RNN(GRUCell(num_cells, kernel_regularizer=None, bias_regularizer=None, recurrent_regularizer=None),
                      return_state=True)
        encoder_outputs, encoder_states = encoder(encoder_inputs)

        decoder_inputs = Input(shape=(None, self.character_set_size))
        decoder_lstm = RNN(GRUCell(num_cells, kernel_regularizer=None, bias_regularizer=None, recurrent_regularizer=None),
                           return_sequences=True, return_state=True)
        decoder_outputs, decoder_state = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.character_set_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        loss = CategoricalCrossentropy()
        optimizer = RMSprop()

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        print(model.summary())
        history = model.fit(x=[self.text_data_train, self.summary_data_train], y=self.summary_data_target_train,
                  batch_size=10,
                  epochs=10, validation_split=0.2)

        model.save(self.model_name)

        pandas.DataFrame(history.history).plot()
        plt.savefig('model.png')
        plt.show()
        score = model.evaluate(x=[self.text_data_test, self.summary_data_test], y=self.summary_data_target_test)
        print(score)

    def decode_samples(self):
        model = load_model(self.model_name)
        predictions = model.predict(self.text_data_test)


if __name__ == '__main__':
    summarization = SummarizationTrial()
    summarization.read_data()
    summarization.create_encoder_decoder()


