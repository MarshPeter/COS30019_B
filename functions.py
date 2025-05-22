import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
from algorithms.Graph import Graph

# Suppress potential warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

def predict_flow_per_scats_sequential(model_path, data_path, lags, hour):
    model = load_model(
        model_path,
        custom_objects={
            'mse': tf.keras.losses.MeanSquaredError(),
            'mape': tf.keras.metrics.MeanAbsolutePercentageError()
        }
    )

    # Process data
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df['hour'] = df['datetime'].dt.hour

    # Sort data by SCATS Number and datetime to ensure correct sequencing
    df = df.sort_values(by=['SCATS Number', 'datetime'])

    # Prepare the scaler
    # Fit the scaler on the entire 'Flow (Veh/hr)' column from the training data
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df['Flow (Veh/hr)'].values.reshape(-1, 1))

    predicted_flows = {}
    # Group by SCATS number to process each one individually
    grouped_scats = df.groupby('SCATS Number')

    for scats_number, scats_data in grouped_scats:
        # For debugging
        # print(f"\nProcessing SCATS Number: {scats_number}")

        # Filter data for the target hour. We need the target hour and 'lags' previous hours.
        # We'll need to iterate through the data to find sequences ending at the target hour.

        # We need to create sequences of 'lags' length for each instance of the target 'hour'.
        # For prediction, we'll use the *latest* available sequence ending at the target hour
        # for each SCATS number.

        scats_data_values = scats_data['Flow (Veh/hr)'].values.reshape(-1, 1)
        scaled_scats_data = scaler.transform(scats_data_values)

        # Find the sequence ending at the *latest* instance of the target hour
        latest_sequence = None
        latest_datetime = None

        # Iterate through the data points for the current SCATS number
        for i in range(len(scats_data) - lags + 1):
            current_sequence = scaled_scats_data[i : i + lags]
            current_datetime = scats_data.iloc[i + lags - 1]['datetime'] # Datetime of the last element in the sequence

            if current_datetime.hour == hour:
                latest_sequence = current_sequence
                latest_datetime = current_datetime

        if latest_sequence is None:
            print(f"Could not find a complete sequence of length {lags} ending at hour {hour} for SCATS {scats_number}.")
            predicted_flows[scats_number] = np.nan # Or some other indicator of no prediction
            continue

        # print(f"Using latest sequence ending at {latest_datetime} for prediction.")

        # Prepare the model input: reshape to (1, lags, 1)
        model_input = latest_sequence.reshape(1, lags, 1)

        # Make prediction
        scaled_prediction = model.predict(model_input, verbose=0)[0][0]

        # Inverse transform the prediction
        predicted_flow = scaler.inverse_transform([[scaled_prediction]])[0][0]

        predicted_flows[scats_number] = predicted_flow
    # Uncomment the following for debugging purposes
    #     print(f"Predicted flow for SCATS {scats_number} at {hour}:00: {predicted_flow:.2f}")

    # print("\nFinal Predictions:")
    # print(predicted_flows)
    return predicted_flows

def create_graph(predicted_flows, nodes_file, edges_file):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)

    graph = Graph()
    for _, row in nodes.iterrows():
        graph.add_node(row['SCAT Number'], row['LATITUDE'], row['LONGITUDE'])

    for _, row in edges.iterrows():
        graph.add_neighbor(int(row['SCATS_A']), int(row['SCATS_B']), float(predicted_flows[row['SCATS_B']]))
        graph.add_neighbor(int(row['SCATS_B']), int(row['SCATS_A']), float(predicted_flows[row['SCATS_A']]))

    return graph
