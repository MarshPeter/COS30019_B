import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. Create the Graph with Longitude and Latitude ---
def create_sample_graph_geo():
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
    locations_plot = {loc: (lon, lat) for loc, (lon, lat) in locations_geo.items()}

    for loc, pos in locations_plot.items():
        G.add_node(loc, pos=pos)

    # Add edges with weights (distances - you might want to calculate these based on geo distance)
    edges = [
        ('A', 'B', 5),  # Example: Assume some distance
        ('A', 'C', 3),
        ('B', 'D', 2),
        ('C', 'D', 1),
        ('C', 'E', 4),
        ('D', 'E', 3)
    ]
    G.add_weighted_edges_from(edges)
    return G, locations_plot # Return plotting coordinates

# --- Rest of your Streamlit App remains similar ---
# ... (calculate_weighted_distance function remains the same)

st.title("Geographical Graph and Distance Calculator")

graph, locations_plot = create_sample_graph_geo() # Use the geo graph function

# --- Display the Graph ---
st.subheader("Graph Visualization (Geographical)")

pos = nx.get_node_attributes(graph, 'pos')
fig, ax = plt.subplots()
nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, ax=ax)
labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='red', ax=ax)

# Optional: Set axis labels to indicate Longitude and Latitude
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Relative Geographical Positions")

st.pyplot(fig)

# ... (User Input and Calculate Distance sections remain the same)
st.subheader("Select Locations")

available_locations = list(graph.nodes())

start_location = st.selectbox("Select Start Location:", available_locations)
end_location = st.selectbox("Select End Location:", available_locations)

if st.button("Calculate Distance"):
    if start_location == end_location:
        st.warning("Start and end locations are the same. Distance is 0.")
    else:
        distance = calculate_weighted_distance(graph, start_location, end_location)
        st.subheader("Total Weighted Distance:")
        st.write(distance)

