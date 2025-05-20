import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings

from Graph import Graph
from AStar import AStar
from DFS import DepthFirst
from Time import calculate_time

warnings.filterwarnings("ignore", category=UserWarning)

def predict_flow_per_scats(model_path, data_path, lags, hour):
    model = load_model(model_path)

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df['hour'] = df['datetime'].dt.hour

    df_hour = df[df['hour'] == hour].copy()
    if df_hour.empty:
        print(f"No data found for {hour}:00 in the training file.")
        return {}

    avg_flows = df_hour.groupby('SCATS Number')['Flow (Veh/hr)'].mean()
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df['Flow (Veh/hr)'].values.reshape(-1, 1))

    predicted_flows = {}
    for scats_number, avg_flow in avg_flows.items():
        scaled_flow = scaler.transform([[avg_flow]])[0][0]
        input_seq = np.array([scaled_flow] * lags).reshape(1, lags, 1)
        scaled_pred = model.predict(input_seq, verbose=0)[0][0]
        predicted_flow = scaler.inverse_transform([[scaled_pred]])[0][0]

        # ✅ Verbose output like your friend's version
        print(f"\nProcessing SCATS Number: {scats_number}")
        print(f"Average {hour}:00 flow for SCATS {scats_number}: {avg_flow:.2f}")
        print(f"Predicted flow for SCATS {scats_number} at {hour}:00: {predicted_flow:.2f}")

        predicted_flows[scats_number] = predicted_flow

    return predicted_flows

def create_graph(predicted_flows, nodes_file, edges_file):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)

    graph = Graph()
    for _, row in nodes.iterrows():
        graph.add_node(row['SCAT Number'], row['LATITUDE'], row['LONGITUDE'])

    for _, row in edges.iterrows():
        flow = predicted_flows.get(row['SCATS_B'], 500)
        graph.add_neighbor(row['SCATS_A'], row['SCATS_B'], flow)
        graph.add_neighbor(row['SCATS_B'], row['SCATS_A'], flow)

    return graph

def main():
    model_file = 'models/lstm_model.h5'
    train_data = './data/train.csv'
    nodes_file = './data/nodes.csv'
    edges_file = './data/intersection_edges.csv'

    lag = 12
    hour = 11
    origin = 970
    destination = 4057

    predicted_flows = predict_flow_per_scats(model_file, train_data, lag, hour)

    # ✅ Print clean final flow dictionary
    clean_flows = {k: float(v) for k, v in predicted_flows.items()}
    print("\nPredicted Flows (LSTM Output):")
    print(clean_flows)

    graph = create_graph(predicted_flows, nodes_file, edges_file)
    graph.set_origin(origin)
    graph.set_goals([destination])

    print("\n=== A* Search ===")
    astar_solver = AStar(graph, predicted_flows)
    print(astar_solver.astar())

    print("\n=== DFS Search ===")
    dfs_solver = DepthFirst(graph)
    print(dfs_solver.dfs())

if __name__ == "__main__":
    main()
