# import pandas as pd

# # define the list of columns to keep
# columns_to_keep = [1, 2, 4, 5, 6, 11, 15, 19,
#                    29, 33, 34, 35, 40, 46, 48, 58, 59, 62, 66]
# columns_to_keep = [x - 1 for x in columns_to_keep]

# # read in the original csv file
# df = pd.read_csv(r"D:\Stuff\CyberSec\Datasets\IDS2018\02-14-2018.csv")

# # select the columns to keep
# df = df.iloc[:, columns_to_keep]

# # convert the label column to numeric values
# label_col = df.columns[-1]
# label_map = {val: i for i, val in enumerate(df[label_col].unique())}
# df[label_col] = df[label_col].map(label_map)

# # take a random sample of 10,000 rows
# df = df.sample(n=10000)

# # export the processed csv file
# df.to_csv('processed.csv', index=False)


# import pandas as pd

# # define the list of columns to keep
# # columns_to_keep = [1,2,4,5,6,11,15,19,29,33,34,35,40,46,48,58,59,62,66]
# columns_to_keep = [1, 2, 4, 5, 6, 11, 15, 19,
#                    29, 33, 34, 35, 40, 46, 48, 58, 59, 62, 66]
# columns_to_keep = [x - 1 for x in columns_to_keep]


# # read in the original csv file
# df = pd.read_csv(r"D:\Stuff\CyberSec\Datasets\IDS2018\02-14-2018.csv")

# # select the columns to keep, including the last label column
# columns_to_keep.append(-1)  # add the last column to the list
# df = df.iloc[:, columns_to_keep]

# # convert the label column to numeric values
# label_col = df.columns[-1]
# label_map = {val: i for i, val in enumerate(df[label_col].unique())}
# df[label_col] = df[label_col].map(label_map)

# # take a random sample of 10,000 rows
# # df = df.sample(n=50000)

# # export the processed csv file
# df.to_csv('processed.csv', index=False)


# import pandas as pd

# # read in the original csv file
# df = pd.read_csv(r"D:\Stuff\CyberSec\Datasets\IDS2018\02-14-2018.csv")

# # remove the 3rd column
# df = df.drop(df.columns[2], axis=1)

# # convert the label column to numeric values
# label_col = df.columns[-1]
# label_map = {val: i for i, val in enumerate(df[label_col].unique())}
# df[label_col] = df[label_col].map(label_map)

# # export the processed csv file
# df.to_csv('processed.csv', index=False)

# # CheckPoint

# import pandas as pd
# import numpy as np

# name = "02-28-2018"
# # read in the original csv file
# df = pd.read_csv(
#     rf"D:\Stuff\CyberSec\Datasets\IDS2018\{name}.csv")

# # convert string labels to integer labels
# if 'label' in df.columns:
#     label_col = 'label'
# elif 'Label' in df.columns:
#     label_col = 'Label'
# else:
#     raise ValueError('Label column not found')
# label_map = {val: i for i, val in enumerate(df[label_col].unique())}
# df[label_col] = df[label_col].map(label_map)

# # remove columns containing alphabets
# df = df.select_dtypes(exclude=['object'])


# # drop columns where all the values are zero
# df = df.loc[:, (df != 0).any(axis=0)]

# # drop columns where each value in the column is zero
# df = df.loc[:, (df != 0).all(axis=0) == False]

# # remove columns where all values are infinite
# df = df.loc[:, (df != np.inf).any(axis=0)]
# df = df.loc[:, (df != -np.inf).any(axis=0)]

# # replace infinite and too large values with NaN
# df = df.replace([np.inf, np.nan, np.finfo(np.float32).max], np.nan)

# # remove rows with null values
# df = df.dropna()

# # sample 10000 rows
# # df = df.sample(n=50000)

# # export the processed csv file
# df.to_csv(
#     rf"D:\Stuff\CyberSec\Datasets\IDS2018\{name}_processed.csv", index=False)

import pandas as pd
import numpy as np

name = "02-20-2018"
# read in the original csv file
df = pd.read_csv(
    rf"D:\Stuff\CyberSec\Datasets\IDS2018\{name}.csv")

# convert string labels to integer labels
if 'label' in df.columns:
    label_col = 'label'
elif 'Label' in df.columns:
    label_col = 'Label'
else:
    raise ValueError('Label column not found')
label_map = {val: i for i, val in enumerate(df[label_col].unique())}
df[label_col] = df[label_col].map(label_map)

# convert object columns to numeric, remove them if the conversion fails
for col in df.select_dtypes(include='object').columns:
    converted_col = pd.to_numeric(df[col], errors='coerce')
    if converted_col.isna().all():
        # all values in the column are non-numeric, drop the column
        df = df.drop(columns=[col])
    else:
        # some values in the column are numeric, replace the column with the converted values
        df[col] = converted_col

# remove rows with non-numeric values in columns where the majority of values are numeric
for col in df.select_dtypes(include=['int', 'float']).columns:
    converted_col = pd.to_numeric(df[col], errors='coerce')
    if converted_col.notna().sum() / len(converted_col) >= 0.5:
        # the majority of values in the column are numeric, remove rows with non-numeric values
        df = df[converted_col.notna()]

# drop columns where all the values are zero
df = df.loc[:, (df != 0).any(axis=0)]

# drop columns where each value in the column is zero
df = df.loc[:, (df != 0).all(axis=0) == False]

# remove columns where all values are infinite
df = df.loc[:, (df != np.inf).any(axis=0)]
df = df.loc[:, (df != -np.inf).any(axis=0)]

# replace infinite and too large values with NaN
df = df.replace([np.inf, np.nan, np.finfo(np.float32).max], np.nan)

# remove rows with null values
df = df.dropna()

# sample 10000 rows
# df = df.sample(n=50000)

# export the processed csv file
df.to_csv(
    rf"D:\Stuff\CyberSec\Datasets\IDS2018\{name}_processed.csv", index=False)
