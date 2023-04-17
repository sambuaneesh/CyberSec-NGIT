import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
data = pd.read_csv(
    r"D:\Stuff\CyberSec\Datasets\IDS2018\02-20-2018_processed.csv")


# Split the data into features and target variable
X = data.iloc[:, :-1]  # all columns except the last one
y = data.iloc[:, -1]  # the last column

# Perform feature ranking using ANOVA F-test
selector = SelectKBest(f_classif, k=30)
X_new = selector.fit_transform(X, y)

# Get the names of the selected features
feature_names = X.columns[selector.get_support(indices=True)]

# Train a Random Forest classifier on the selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_new, y)

# Evaluate the model performance
accuracy = clf.score(X_new, y)
print("Accuracy:", accuracy)
print("Selected features:", feature_names)
