from tensorflow import keras

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input


def build_model_v1(time_step: int, n_features: int):
    model = Sequential()

    model.add(Input(shape=(time_step, n_features)))

    model.add(LSTM(units=128, activation="tanh", return_sequences=True))

    model.add(LSTM(units=64))

    model.add(Dense(25))
    model.add(Dense(2))

    model.compile(optimizer="adam", loss="mean_squared_error")

    return model
