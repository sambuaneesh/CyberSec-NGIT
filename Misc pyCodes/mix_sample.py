# import os
# import pandas as pd

# # set the directory path where the datasets are located
# dir_path = r'D:\Stuff\CyberSec\Datasets\IDS2018_extract\concat_op'

# # create an empty list to hold the sampled DataFrames
# dfs = []

# # loop through each CSV file in the directory
# for i, file_name in enumerate(os.listdir(dir_path)):
#     if file_name.endswith('.csv'):
#         # read the CSV file into a DataFrame
#         with open(os.path.join(dir_path, file_name)) as file:
#             df = pd.read_csv(file)

#         # sample 10k rows from the DataFrame
#         df_sampled = df.sample(n=10000)

#         # add the sampled DataFrame to the list
#         dfs.append(df_sampled)

#         # close the file
#         file.close()

#         # print progress update
#         print(f'Sampled {i+1} files')

# # combine the sampled DataFrames into a single DataFrame
# combined_df = dfs[0]
# for i in range(1, len(dfs)):
#     combined_df = combined_df.join(dfs[i], rsuffix=f'_{i}')

# # export the combined DataFrame to a CSV file
# combined_df.to_csv(
#     r"D:\Stuff\CyberSec\Datasets\IDS2018_extract\sampled.csv", index=False)

# # print completion message
# print('Data combination complete')
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
    with open(csv_file, "r") as f:
        df = pd.read_csv(f, nrows=40000)  # Sample only 10k rows
        if not column_names:
            # If column_names is empty, add all columns to it
            column_names.update(set(df.columns))
        elif column_names != set(df.columns):
            # If the columns in this file are not the same as in previous files, print the difference
            print(
                f"The columns in {csv_file} do not match the columns in other files.")
            print(f"Columns in {csv_file}: {set(df.columns)}")
            print(f"Columns in previous files: {column_names}")
        # Append the sampled data to a new CSV file, excluding headers for all files except the first one
        df.to_csv('combined_data.csv', mode='a', index=False,
                  header=not os.path.exists('combined_data.csv'))

print("Combination complete.")
