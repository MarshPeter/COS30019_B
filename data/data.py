import pandas as pd
import numpy as np
import os

# speed_limit = 60
# coef_a = -1.4648375
# coef_b = 93.75
# coefficients = [coef_a, coef_b, 0]
# roots = np.roots(coefficients)
# flow_maximum_speed = (roots[0] + roots[1]) / 2
# print(roots)
# print(flow_maximum_speed)

csv_file = "./ScatsDataOctober2006.csv"
new_csv_file = "./train.csv"

df = pd.read_csv(csv_file)

new_rows_list = []
# new_df = pd.DataFrame(columns=['SCATS Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE', 'NB_LONGITUDE', 'datetime', 'Flow (Veh/hr)'])

for index, row in df.iterrows():
    # print(row)
    # Disgusting for loop because I'm a brainlet
    hour = 0
    for i in range(0, 96, 4):
        timestamp1 = int(row[f'V0{i}'] if i < 10 else row[f'V{i}'])
        timestamp2 = int(row[f'V0{i + 1}'] if i + 1 < 10 else row[f'V{i + 1}'])
        timestamp3 = int(row[f'V0{i + 2}'] if i + 2 < 10 else row[f'V{i + 2}'])
        timestamp4 = int(row[f'V0{i + 3}'] if i + 3 < 10 else row[f'V{i + 3}'])

        hour_flow_total = timestamp1 + timestamp2 + timestamp3 + timestamp4
        new_row_data = {
            'SCATS Number': row['SCATS Number'],
            'Location': row['Location'],
            'CD_MELWAY': row['CD_MELWAY'],
            'NB_LATITUDE': row['NB_LATITUDE'],
            'NB_LONGITUDE': row['NB_LONGITUDE'],
            # Assuming 'Date' is a string. If it's a datetime object, you might need to format it first.
            'datetime': str(row['Date']) + f" {hour}:00",
            'Flow (Veh/hr)': hour_flow_total
        }

        new_rows_list.append(new_row_data)
        hour += 1

new_df = pd.DataFrame(new_rows_list)

print(df.head())
print(new_df.tail())

new_df.to_csv(new_csv_file)
