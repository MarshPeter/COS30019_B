import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from functions import predict_flow_per_scats_sequential, create_graph
from algorithms.UniformCost import UniformCost
from algorithms.bfs import BFS
from algorithms.greedy import Greedy
from algorithms.DFS import DepthFirst
from algorithms.AStar import AStar

# --- 1. Create the Graph with Longitude and Latitude ---
def create_sample_graph_geo(node_data, edges_data, flows_data):
    G = nx.Graph()
    # Add nodes with longitude and latitude
    locations_geo = {
        'A': (-74.0060, 40.7128),  # Example: New York City
        'B': (-73.9352, 40.7922),  # Example: Central Park
        'C': (-74.0045, 40.7420),  # Example: Greenwich Village
        'D': (-73.9856, 40.7580),  # Example: Times Square
        'E': (-73.9555, 40.7820)   # Example: Upper East Side
    }
    # Map geo coordinates to plotting coordinates (simple Plate Carrée)
    # locations_plot = {loc: (lon, lat) for loc, (lon, lat) in locations_geo.items()}

    locations_plot = {}

    for _, row in node_data.iterrows():
        locations_plot[int(row['SCAT Number'])] = (row['LONGITUDE'], row['LATITUDE'])

    for loc, pos in locations_plot.items():
        G.add_node(loc, pos=pos)

    edges = []

    for _, row in edges_data.iterrows():
        edges.append((int(row['SCATS_A']), int(row['SCATS_B']), flows_data.loc[row['SCATS_B'], 'Flow (Veh/hr)']))
        edges.append((int(row['SCATS_B']), int(row['SCATS_A']), flows_data.loc[row['SCATS_A'], 'Flow (Veh/hr)']))

    # Add edges with weights (distances - you might want to calculate these based on geo distance)
    # edges = [
    #     ('A', 'B', 5),  # Example: Assume some distance
    #     ('A', 'C', 3),
    #     ('B', 'D', 2),
    #     ('C', 'D', 1),
    #     ('C', 'E', 4),
    #     ('D', 'E', 3)
    # ]
    G.add_weighted_edges_from(edges)

    return G, locations_plot # Return plotting coordinates

def main(nodes_csv, edges_csv, flows_csv):
    node_data = pd.read_csv(nodes_csv)
    edges_data = pd.read_csv(edges_csv)
    flows_data = pd.read_csv(flows_csv)

    st.title("Geographical Graph and Distance Calculator")

    graph, locations_plot = create_sample_graph_geo(node_data, edges_data, flows_data) # Use the geo graph function

    # --- Display the Graph ---
    st.subheader("Graph Visualization (Geographical)")

    pos = nx.get_node_attributes(graph, 'pos')
    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw(graph, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=6, ax=ax)
    labels = nx.get_edge_attributes(graph, 'weight')

    ax.set_aspect('equal')
    # Optional: Set axis labels to indicate Longitude and Latitude
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Relative Geographical Positions")

    st.pyplot(fig)

    # ... (User Input and Calculate Distance sections remain the same)
    st.subheader("Select Locations")

    available_locations = list(graph.nodes())

    start_location = st.text_input("Select Start Location:")
    end_location = st.text_input("Select End Location:")
    model_choice = st.selectbox("Select desired ML model: ", ['sRNN', 'LSTM', 'GRU'])
    algorithm_choice = st.selectbox("Select desired graphing algorithm: ", ['BFS', 'DFS', 'Uniform Cost', 'Greedy', 'AStar'])
    hour = st.text_input("Select hour 0-23 (representing 24 hour time)")


    if st.button("CalculatePath"):
        if start_location == end_location:
            st.warning("Start and end locations are the same. Time is 0.")
        elif 0 > int(hour) or 23 < int(hour):
            st.warning("You  need a valid hour")
        else:
            result = get_path(model_choice, algorithm_choice, int(hour), int(start_location), int(end_location))

            if algorithm_choice == "BFS":
                print(result)
                path = result[0][1][0]
                time = result[0][1][1]
                st.write(f"path: {path}")
                st.write(f"Time: {time} minutes")
            elif algorithm_choice == "DFS":
                print(result)
                path = result[0][1][0]
                time = result[0][1][1]
                st.write(f"path: {path}")
                st.write(f"Time: {time} minutes")
            elif algorithm_choice == "Uniform Cost":
                print(result)
                path = result[0][1][0]
                time = result[0][1][1]
                st.write(f"path: {path}")
                st.write(f"Time: {time} minutes")
            elif algorithm_choice == "Greedy":
                print(result)
                path = result[0][1]
                time = result[0][2]
                st.write(f"path: {path}")
                st.write(f"Time: {time} minutes")
            elif algorithm_choice == "AStar":
                print(result)
                path = result[0][1][0]
                time = result[0][1][1]
                st.write(f"path: {path}")
                st.write(f"Time: {time} minutes")

def get_path(model, algorithm, hour, origin, destination):
    model_file = f'./models/{model}.h5'  
    train_data_file = './data/train.csv'
    edges_file = './data/edges.csv'
    nodes_file = "./data/nodes.csv"
    lag_value = 12  # Matches training lag

    predicted_flows = predict_flow_per_scats_sequential(model_file, train_data_file, lag_value, hour)
    graph = create_graph(predicted_flows, nodes_file, edges_file)
    graph.set_origin(origin)
    graph.set_goals([destination]) # legacy requirement to have goals be in lists    

    if algorithm == "BFS":
        solution = BFS(graph)
        return solution.breadth_first_search()
    elif algorithm == "DFS":
        solution = DepthFirst(graph)
        return solution.dfs()
    elif algorithm == "Uniform Cost":
        solution = UniformCost(graph)
        return solution.uniform_cost_search()
    elif algorithm == "Greedy":
        solution = Greedy(graph)
        return solution.gbfs()
    elif algorithm == "AStar":
        solution = AStar(graph)
        return solution.astar()

if __name__ == "__main__":
    nodes_file = "./data/nodes.csv"
    edges_file = "./data/edges.csv"
    flows_file = "./data/train.csv"
    main(nodes_file, edges_file, flows_file)
