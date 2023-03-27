import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from CSV file
df = pd.read_csv(r"D:\Stuff\CyberSec\archive\02-15-2018.csv")

# Remove any rows with missing values
df = df.dropna()

# Drop columns where all values are 0
df = df.loc[:, (df != 0).any(axis=0)]

for start_col in range(0, 69, 1):
    end_col = start_col + 1
    try:
        X = df.iloc[:, start_col:end_col].values
        y = df.iloc[:, -1].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Train the decision tree classifier
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        # Test the classifier
        accuracy = clf.score(X_test, y_test)
        print(f"Accuracy for columns {start_col}:{end_col}: {accuracy}")
    except:
        print(f"Error with columns {start_col}:{end_col}")
