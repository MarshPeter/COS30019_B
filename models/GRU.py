import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.models import Sequential

def get_GRU(units):
    
    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model