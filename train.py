import pandas as pd
import numpy as np
from models.sRNN import get_sRNN
from LSTMModel import build_lstm_model
from sklearn.preprocessing import MinMaxScaler

def process_data(lags):
    training_data = "./data/train.csv"   
    df = pd.read_csv(training_data)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df['hour'] = df['datetime'].dt.hour

    print(df.head())
    attr = "Flow (Veh/hr)"
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df[attr].values.reshape(-1, 1))
    flow = scaler.transform(df[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    print(df.head())

    train = []

    for i in range(lags, len(flow)):
        train.append(flow[i - lags: i+1])

    train = np.array(train)
    np.random.shuffle(train)

    return train[:, :-1], train[:, -1], scaler

def train_model(model, X_train, y_train, name, config):
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('models/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('models/' + name + ' loss.csv', encoding='utf-8', index=False)

def main():
    lag = 12
    config = {"batch": 256, "epochs": 600}

    X_train, Y_train, _ = process_data(lag)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    m = build_lstm_model([12, 64, 64, 1])
    train_model(m, X_train, Y_train, "LSTM", config)

if __name__ == "__main__":
    main()
