import os
import pandas as pd

# Directory containing the datasets
data_dir = r"D:\Stuff\CyberSec\Datasets\IDS2018_extract\concat_op"

# List all the csv files in the directory
csv_files = [os.path.join(data_dir, f)
             for f in os.listdir(data_dir) if f.endswith(".csv")]

# Set to store column names
column_names = set()

# Loop through each csv file, read in the data, and add the column names to the set
for csv_file in csv_files:
    print(f"Reading in data from {csv_file}...")
    with open(csv_file, "r") as f:
        df = pd.read_csv(f)
        column_names.update(set(df.columns))

# Check if the intersection of the sets is equal to the set of column names of any one dataset
if len(column_names.intersection(set(df.columns))) == len(set(df.columns)):
    print("All datasets have the same columns.")
else:
    print("Not all datasets have the same columns.")
