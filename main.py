import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings

from algorithms.Graph import Graph
from algorithms.UniformCost import UniformCost

# Suppress potential warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

def predict_flow_per_scats(model_path, data_path, lags, hour):
    model = load_model(
        model_path,
        custom_objects={
            # These are to prevent errors
            'mse': tf.keras.losses.MeanSquaredError(),
            'mape': tf.keras.metrics.MeanAbsolutePercentageError()
        }
    )

    # Process training data
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df['hour'] = df['datetime'].dt.hour

    # Filter for all entries of the specified hour
    eleven_am_df = df[df['hour'] == hour].copy() 

    if eleven_am_df.empty:
        print(f"No data found for ${hour}:00 in the training file.")
        return {}

    # Calculate the average hour flow for each SCATS number
    average_eleven_am_per_scats = eleven_am_df.groupby('SCATS Number')['Flow (Veh/hr)'].mean()

    # Prepare the scaler so that we can actually use it as test data
    # Fit the scaler on the entire 'Flow (Veh/hr)' column from the training data
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df['Flow (Veh/hr)'].values.reshape(-1, 1))

    predicted_flows = {}
    # Iterate through each SCATS number and make predictions
    for scats_number, avg_flow in average_eleven_am_per_scats.items():
        print(f"\nProcessing SCATS Number: {scats_number}")
        print(f"Average 11:00 AM flow for SCATS {scats_number}: {avg_flow:.2f}")

        # Scale the average hour flow for the current SCATS number
        scaled_average_flow = scaler.transform([[avg_flow]])[0][0]

        # Create the model input
        model_input = np.array([scaled_average_flow] * lags).reshape(1, lags, 1)
        # print(f"Model input shape for SCATS {scats_number}: {model_input.shape}") # Optional: uncomment for debugging

        # Make prediction
        scaled_prediction = model.predict(model_input, verbose=0)[0][0] # verbose=0 to reduce output

        # Inverse transform the prediction
        predicted_flow = scaler.inverse_transform([[scaled_prediction]])[0][0]

        predicted_flows[scats_number] = predicted_flow
        print(f"Predicted flow for SCATS {scats_number} at 11:00 AM: {predicted_flow:.2f}")

    print(predicted_flows)
    return predicted_flows

def create_graph(predicted_flows, nodes_file, intersections_file):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(intersections_file)

    graph = Graph()
    for _, row in nodes.iterrows():
        graph.add_node(row['SCAT Number'], row['LATITUDE'], row['LONGITUDE'])

    for _, row in edges.iterrows():
        graph.add_neighbor(row['SCATS_A'], row['SCATS_B'], float(predicted_flows[row['SCATS_B']]))
        graph.add_neighbor(row['SCATS_B'], row['SCATS_A'], float(predicted_flows[row['SCATS_A']]))

    return graph

def main():
    model_file = 'models/sRNN.h5'  
    train_data_file = './data/train.csv'
    intersections_file  = './data/intersection_edges.csv'
    nodes_file = "./data/nodes.csv"
    lag_value = 12  # Matches training lag
    hour = 11
    origin = 970
    destination = 4057

    predicted_flows = predict_flow_per_scats(model_file, train_data_file, lag_value, hour)
    graph = create_graph(predicted_flows, nodes_file, intersections_file)
    graph.set_origin(origin)
    graph.set_goals([destination]) # legacy requirement to have goals be in lists

    solution = UniformCost(graph)

    print(solution.uniform_cost_search())

if __name__ == "__main__":
    main()
