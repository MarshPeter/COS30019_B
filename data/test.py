import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'non_aggregated.csv'

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Select the relevant columns and drop duplicate rows to get unique combinations
    unique_locations = df[['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()

    # Print the result
    print(unique_locations)

    unique_locations.to_csv("./test2.csv", index=False)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
