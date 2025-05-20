import pandas as pd
import numpy as np
from LSTMModel import train_lstm_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def process_data(lags):
    df = pd.read_csv("./data/train.csv")
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df['hour'] = df['datetime'].dt.hour

    attr = "Flow (Veh/hr)"
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df[attr].values.reshape(-1, 1))
    flow = scaler.transform(df[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train = []
    for i in range(lags, len(flow)):
        train.append(flow[i - lags: i+1])

    train = np.array(train)
    np.random.shuffle(train)

    return train[:, :-1], train[:, -1], scaler

def main():
    lag = 12
    X_train, Y_train, _ = process_data(lag)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model, history = train_lstm_model(X_train, Y_train, X_train, Y_train, epochs=100, batch_size=64)
    model.save("models/lstm_model.h5")

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv("models/lstm_loss.csv", index=False)
    print("LSTM model training complete. Files saved in models/")

if __name__ == "__main__":
    main()