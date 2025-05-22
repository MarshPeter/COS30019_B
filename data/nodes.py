from collections import defaultdict
import pandas as pd
import networkx as nx
from geopy.distance import geodesic

# We use this one to figure out the longitudes of each SCAT Point

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

def create_estimated_locations(data_file, save_file):
    non_aggregated_csv_df = pd.read_csv(data_file)

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
    # df_edges['distance (km)'] = df_edges.apply(
    #     lambda row: geodesic(
    #         (centroids[row['SCATS_A']][1], centroids[row['SCATS_A']][0]),
    #         (centroids[row['SCATS_B']][1], centroids[row['SCATS_B']][0]),
    #     ).km,
    #     axis=1
    # )

    # Information about the approximate location of all nodes
    node_data = []
    for scat_number, coords in centroids.items():
        node_data.append({
            "SCAT Number": scat_number,
            "LONGITUDE": coords[0],
            "LATITUDE": coords[1]
        })

    node_df = pd.DataFrame(node_data)

    node_df.to_csv(save_file, index=False)

def main():
    non_aggregated_csv = "./non_aggregated.csv"
    nodes_csv = "./nodes.csv"
    create_estimated_locations(non_aggregated_csv, nodes_csv)


if __name__ == "__main__":
    main()

