from collections import defaultdict
import pandas as pd
import networkx as nx

from_csv = "./locations.csv"
to_csv = "./edges.csv"

df = pd.read_csv("./locations.csv")

opp = {"N":"S", "S":"N", "E":"W", "W":"E",
       "NE":"SW","SW":"NE","NW":"SE","SE":"NW"}

Scats = defaultdict(list)
for index, row in df.iterrows():
    values = row["Location"].split()
    already_in_data = False
    if row["SCATS"] in Scats:
        for v in Scats[row["SCATS"]]:
            if values[0] == v[0] and values[1] == v[1]:
                already_in_data = True
                break

    if not already_in_data:
        Scats[row["SCATS"]].append((values[0], values[1]))
        
raw_data_scats = dict(Scats)        

# 2) opposite‐direction map
opp = {"N":"S", "S":"N", "E":"W", "W":"E",
       "NE":"SW","SW":"NE","NW":"SE","SE":"NW"}

# 3) build lookup table
lookup = {}
for node, segs in raw_data_scats.items():
    for road, d in segs:
        lookup.setdefault((road, d), []).append(node)

# 4) build the graph
G = nx.Graph()
for node in raw_data_scats:
    G.add_node(node)

for node, segs in raw_data_scats.items():
    for road, d in segs:
        od = opp[d]
        for nbr in lookup.get((road, od), []):
            if nbr == node:
                continue
            # add an edge labelled by the road name
            G.add_edge(node, nbr, road=road)

# now G.nodes is your list of intersections
# and G.edges gives you the directly‐connected pairs
for u, v, data in G.edges(data=True):
    print(f"{u} ⟷ {v} via {data['road']}")

df_edges = nx.to_pandas_edgelist(G)

# optionally rename the columns to whatever you like
df_edges = df_edges.rename(
    columns={
        "source": "SCATS_A",
        "target": "SCATS_B",
        "road":   "road_name",
    }
)

# write it out
df_edges.to_csv("intersection_edges.csv", index=False)
