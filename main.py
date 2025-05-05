import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph (since traffic flow is directional)
G = nx.DiGraph()

# Add nodes (intersections)
G.add_node("A", pos=(0, 0))
G.add_node("B", pos=(1, 1))
G.add_node("C", pos=(2, 0))
G.add_node("D", pos=(1, -1))
G.add_node("E", pos=(3, 1))

# Add edges (roads) with predicted travel times (weights)
G.add_edge("A", "B", weight=5)  # 5 minutes
G.add_edge("A", "D", weight=10) # 10 minutes
G.add_edge("B", "C", weight=7)  # 7 minutes
G.add_edge("B", "E", weight=8)  # 8 minutes
G.add_edge("C", "E", weight=4)  # 4 minutes
G.add_edge("D", "B", weight=6)  # 6 minutes
G.add_edge("D", "C", weight=9)  # 9 minutes

# Get positions for drawing
pos = nx.get_node_attributes(G, 'pos')

plt.figure(figsize=(8, 6))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20, edge_color='gray', width=2)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# Draw edge labels (weights)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Weighted Traffic Network Graph (Matplotlib)")
plt.axis('off') # Hide axes
plt.show()
