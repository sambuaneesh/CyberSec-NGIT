import pandas as pd
import numpy as np

# read in the original csv file
data = pd.read_csv(r"D:\Stuff\CyberSec\Notebooks\generated_data.csv")

mode_column2 = data["Dst Port"].mode()[0]
data["Dst Port"] = data["Dst Port"].fillna(mode_column2).astype(int)


# Define the range of columns to convert
start_col = 'Protocol'
end_col = 'Fwd Pkt Len Min'

# Iterate over the columns in the range and convert them to integers
for col in data.loc[:, start_col:end_col]:
    data[col] = data[col].fillna(data[col].median()).round().astype(int)

data.to_csv('modified_data.csv', index=False)


# import pandas as pd

# # Import the CSV file into a pandas dataframe
# data = pd.read_csv('original_data.csv')

# # Calculate the mean for column "Column1"
# mean_column1 = data["Column1"].mean()

# # Round column "Column1" to the nearest integer
# data["Column1"] = data["Column1"].round()

# # Convert column "Column2" to integer using the mode value
# mode_column2 = data["Column2"].mode()[0]
# data["Column2"] = data["Column2"].fillna(mode_column2).astype(int)

# # Convert column "Column3" to integer using the median value
# median_column3 = data["Column3"].median()
# data["Column3"] = data["Column3"].fillna(median_column3).round().astype(int)

# # Save the resulting dataframe to a CSV file
# data.to_csv('modified_data.csv', index=False)
