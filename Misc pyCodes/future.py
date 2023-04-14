# Import necessary libraries
import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
data = pd.read_csv('input_data.csv')

# Define a list of column names to approximate
cols_to_approximate = ['Column1', 'Column2', 'Column3']

# Loop over each column and approximate using mean, median or mode
for col in cols_to_approximate:
    if data[col].dtype == 'object':
        # For categorical columns, use mode
        mode_val = data[col].mode()[0]
        data[col] = data[col].fillna(mode_val)
    else:
        # For numerical columns, use mean or median
        if data[col].isnull().sum() > 0:
            # If there are missing values, use median
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
        else:
            # Otherwise, use mean
            mean_val = data[col].mean()
            data[col] = data[col].fillna(mean_val)

# Round the numerical columns to two decimal places
num_cols = data.select_dtypes(include=[np.number]).columns
data[num_cols] = data[num_cols].round(decimals=2)

# Save the modified DataFrame to a new CSV file
data.to_csv('output_data.csv', index=False)
