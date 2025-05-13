from collections import defaultdict
import pandas as pd
import networkx as nx
from geopy.distance import geodesic

# We use this one to figure out the longitudes of each SCAT Point
non_aggregated_csv = "./non_aggregated.csv"
from_csv = "./locations.csv"
to_csv = "./edges.csv"
nodes_csv = "./nodes.csv"

def calculate_centroid(points):
    if not points:
        return None

    sum_lon = 0
    sum_lat = 0

    for lon, lat in points:
        sum_lon += lon
        sum_lat += lat

    cent_lon = sum_lon / len(points)
    cent_lat = sum_lat / len(points)

    return (cent_lon, cent_lat)

def filter_scats_into_centroids(loc_data, center_lon = 145.0, lon_tolerance = 2, center_lat = -37, lat_tolerance = 2):
    filtered_scats = {}
    min_lon = center_lon - lon_tolerance
    max_lon = center_lon + lon_tolerance
    min_lat = center_lat - lat_tolerance
    max_lat = center_lat + lat_tolerance

    for scat_number, data in loc_data.items():
        longitudes = data.get('longitudes', [])
        latitudes = data.get('latitudes', [])

        valid_points = []

        for i in range(len(longitudes)):
            lon = longitudes[i]
            lat = latitudes[i]

            if min_lon < lon <= max_lon and min_lat <= lat <= max_lat:
                valid_points.append((lon, lat))

        centroid = calculate_centroid(valid_points)

        # entered into dictionary as: scat_number: (lon, lat)
        filtered_scats[scat_number] = centroid

    return filtered_scats

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


non_aggregated_csv_df = pd.read_csv(non_aggregated_csv)

already_encountered_scat_locations = list()
scat_location_values = {}

for _, row in non_aggregated_csv_df.iterrows():
    if row["Location"] in already_encountered_scat_locations:
        continue

    already_encountered_scat_locations.append(row["Location"])

    if row["SCATS Number"] not in scat_location_values:
        scat_location_values[row["SCATS Number"]] = {
            "longitudes": [float(row["NB_LONGITUDE"])],
            "latitudes": [float(row["NB_LATITUDE"])],
            "count": 1
        }
    else:
        if row["Location"] in scat_location_values[row["SCATS Number"]]:
            continue
        scat_location_values[row["SCATS Number"]]["longitudes"].append(row["NB_LONGITUDE"])
        scat_location_values[row["SCATS Number"]]["latitudes"].append(row["NB_LATITUDE"])
        scat_location_values[row["SCATS Number"]]["count"] = scat_location_values[row["SCATS Number"]]["count"] + 1

# print(already_encountered_scat_locations)
# print(scat_location_values)

# Dictionary is in scat_number: (long, lat)
centroids = filter_scats_into_centroids(scat_location_values)
print(centroids)

# Information about the distance between all edges
df_edges['distance (km)'] = df_edges.apply(
    lambda row: geodesic(
        (centroids[row['SCATS_A']][1], centroids[row['SCATS_A']][0]),
        (centroids[row['SCATS_B']][1], centroids[row['SCATS_B']][0]),
    ).km,
    axis=1
)

# Information about the approximate location of all nodes
node_data = []
for scat_number, coords in centroids.items():
    node_data.append({
        "SCAT Number": scat_number,
        "LONGITUDE": coords[0],
        "LATITUDE": coords[1]
    })

node_df = pd.DataFrame(node_data)

node_df.to_csv(nodes_csv, index=False)

# write it out
df_edges.to_csv("intersection_edges.csv", index=False)

