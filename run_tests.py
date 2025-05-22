import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings

from algorithms.AStar import AStar
from algorithms.DFS import DepthFirst
from algorithms.UniformCost import UniformCost
from algorithms.bfs import BFS
from algorithms.greedy import Greedy
from functions import predict_flow_per_scats_sequential, create_graph

tests = [
    ('sRRN', 'BFS', 4263, 3685, 16),
    ('LSTM', 'BFS', 4057, 4063, 21),
    ('LSTM', 'DFS', 4272, 3682, 19),
    ('LSTM', 'Greedy', 4266, 4266, 18),
    ('LSTM', 'AStar', 4035, 4821, 22),
    ('LSTM', 'AStar', 4032, 3685, 11),
    ('sRRN', 'DFS', 970, 4057, 8),
    ('GRU', 'AStar', 3662, 3180, 16),
    ('sRRN', 'DFS', 3127, 2820, 0),
    ('LSTM', 'Greedy', 4273, 4324, 9),
    ('LSTM', 'AStar', 2000, 4035, 14),
    ('sRRN', 'UniformCost', 3685, 4035, 1),
    ('GRU', 'AStar', 3685, 3127, 20),
    ('sRRN', 'AStar', 4057, 4057, 5),
    ('GRU', 'UniformCost', 3662, 3127, 14),
    ('LSTM', 'UniformCost', 4324, 4030, 19),
    ('GRU', 'UniformCost', 3120, 3126, 15),
    ('GRU', 'AStar', 3812, 3180, 4),
    ('sRRN', 'UniformCost', 4812, 4270, 7),
    ('sRRN', 'DFS', 4321, 4335, 4)
]

def run_test(model_file, algorithm, origin, destination, hour):
    train_data_file = './data/train.csv'
    edges_file = './data/edges.csv'
    nodes_file = "./data/nodes.csv"
    lag_value = 12  # Matches training lag
    predicted_flows = predict_flow_per_scats_sequential(model_file, train_data_file, lag_value, hour)
    graph = create_graph(predicted_flows, nodes_file, edges_file)
    graph.set_origin(origin)
    graph.set_goals([destination]) # legacy requirement to have goals be in lists

    result = None

    if algorithm == "BFS":
        solution = BFS(graph)
        result = solution.breadth_first_search()
    elif algorithm == "DFS":
        solution = DepthFirst(graph)
        result = solution.dfs()
    elif algorithm == "Uniform Cost":
        solution = UniformCost(graph)
        result = solution.uniform_cost_search()
    elif algorithm == "Greedy":
        solution = Greedy(graph)
        result = solution.gbfs()
    elif algorithm == "AStar":
        solution = AStar(graph)
        result = solution.astar()

    if algorithm == "BFS":
        print(result)
        path = result[0][1][0]
        time = result[0][1][1]
        print(f"path: {path}")
        print(f"Time: {time} minutes")
    elif algorithm == "DFS":
        print(result)
        path = result[0][1][0]
        time = result[0][1][1]
        print(f"path: {path}")
        print(f"Time: {time} minutes")
    elif algorithm == "Uniform Cost":
        print(result)
        path = result[0][1][0]
        time = result[0][1][1]
        print(f"path: {path}")
        print(f"Time: {time} minutes")
    elif algorithm == "Greedy":
        print(result)
        path = result[0][1]
        time = result[0][2]
        print(f"path: {path}")
        print(f"Time: {time} minutes")
    elif algorithm == "AStar":
        print(result)
        path = result[0][1][0]
        time = result[0][1][1]
        print(f"path: {path}")
        print(f"Time: {time} minutes") 

def main():
    for test in tests:
        print(f"Test: {test[0]} + {test[1]}, from {test[2]} to {test[3]}, at hour {test[4]}:00")
        run_test(f"models/{test[0]}.h5", test[1], test[2], test[3], test[4])

# NOTE: This main function is for testing in a console environment, not intended for actual demonstration
if __name__ == "__main__":
    main()
