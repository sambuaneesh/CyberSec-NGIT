import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


class IDSDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __getitem__(self, index):
        features = self.data.iloc[index, :-1].values
        features = torch.tensor(features, dtype=torch.float32)
        label = self.data.iloc[index, -1]
        label = torch.tensor(label, dtype=torch.long)
        return features, label

    def __len__(self):
        return len(self.data)


class IDSClassifier(nn.Module):
    def __init__(self):
        super(IDSClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = IDSDataset(r"D:\Stuff\CyberSec\Datasets\IDS2018\02-14-2018.csv")
train_data, val_data = train_test_split(train_data, test_size=0.1)

batch_size = 64
num_epochs = 10

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

model = IDSClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(num_epochs):
    model.train()

    train_loss = 0.0
    train_correct = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * features.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_data)
    train_acc = train_correct.double() / len(train_data)

    model.eval()

    val_loss = 0.0
    val_correct = 0

    for features, labels in val_loader:
        features = features.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(features)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_data)
    val_acc = val_correct.double() / len(val_data)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

test_data = IDSDataset(r"D:\Stuff\CyberSec\Datasets\IDS2018\02-14-2018.csv")
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model.eval()

test_loss = 0.0
test_correct = 0

for features, labels in test_loader:
    features = features.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(features)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * features.size(0)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels.data)

test_loss = test_loss / len(test_data)
test_acc = test_correct.double() / len(test_data)

print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(test_loss, test_acc))

# Generate a classification report for the test set
test_preds = []
test_labels = []

for features, labels in test_loader:
    features = features.to(device)

    with torch.no_grad():
        outputs = model(features)
        _, preds = torch.max(outputs, 1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

print(classification_report(test_labels, test_preds,
      target_names=['benign', 'malicious']))
