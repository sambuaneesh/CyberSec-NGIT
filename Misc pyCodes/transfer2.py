import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load the CSV file using Pandas
df = pd.read_csv(r"D:\Stuff\CyberSec\Datasets\IDS2018\02-14-2018.csv")

# Check for any missing values
print(df.isnull().sum())

# Convert non-numeric data types to numeric data types
df = df.apply(pd.to_numeric, errors='coerce')

# Check the data types of all columns
print(df.dtypes)

# Remove any rows or columns that contain missing values or incompatible data types
df.dropna(inplace=True)
df = df.select_dtypes(include=[np.float32, np.float64, np.int32, np.int64])

# Convert the Pandas DataFrame to a NumPy array and then to a PyTorch tensor
features = df.drop('Label', axis=1).to_numpy()
labels = df['Label'].to_numpy()
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Split the dataset into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    features, labels, test_size=0.2)

# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Freeze all layers in the model
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one that has the correct number of output classes
num_classes = 2  # Change this to the number of output classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Train the model
batch_size = 64
num_epochs = 10
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        inputs = train_data[i:i+batch_size]
        labels = train_labels[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Evaluate the model on the validation set
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            inputs = val_data[i:i+batch_size]
            labels = val_labels[i:i+batch_size]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch +
          1, num_epochs, train_losses[-1], val_losses[-1]))

# Save the trained model
torch.save(model.state_dict(),
           r"D:\Stuff\CyberSec\Datasets\IDS2018\02-14-2018.csv")
