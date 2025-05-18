import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

def get_sRNN(units):
    model = Sequential()
    model.add(SimpleRNN(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(SimpleRNN(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model
