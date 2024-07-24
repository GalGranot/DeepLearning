# Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data
col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
feature_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
#data = pd.read_csv("/c/Users/Galag/Documents/GitHub/DeepLearning/HW2/magic04.data", names=col_names)
data = pd.read_csv("magic04.data", names=col_names)
X = data[feature_names]
Y = data['class']

# Split the data into train, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.66)

# Pre-process and convert labels to integers
def preprocess_labels(labels):
    return np.where(labels == 'g', 0, 1)

x_train = x_train.to_numpy()
y_train = preprocess_labels(y_train)

x_test = x_test.to_numpy()
y_test = preprocess_labels(y_test)

x_validation = x_validation.to_numpy()
y_validation = preprocess_labels(y_validation)

# Train a Logistic Regression model as a baseline
logistic_model = LogisticRegression(solver='lbfgs')
logistic_model.fit(x_train, y_train)
y_pred_train = logistic_model.predict(x_train)
print(f"Number of mislabeled points {np.sum(y_train != y_pred_train)} out of {x_train.shape[0]} total points.")
print(f"Logistic Regression Model accuracy = {logistic_model.score(x_test, y_test)}")

# Create TensorDataset from numpy arrays
tensor_train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))

# Define the neural network model
class MagicClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MagicClassifier, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 4),
        )
        self.output_layer = nn.Linear(4, output_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output_layer(x)
        return x

# Hyperparameters
features_num = len(feature_names)
output_dim = 1
batch_size = 128
learning_rate = 0.001
epochs = 100
criterion = nn.BCEWithLogitsLoss()
model = MagicClassifier(input_dim=features_num, output_dim=output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
tensor_train_dataloader = DataLoader(tensor_train_ds, batch_size=batch_size, shuffle=True)

# Training the model
for epoch in range(epochs):
    model.train()
    losses = []
    for features, targets in tensor_train_dataloader:
        optimizer.zero_grad()
        output = model(features).squeeze()
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses)}")

# Optional: Evaluate on validation set
tensor_validation_ds = TensorDataset(torch.tensor(x_validation, dtype=torch.float32), torch.tensor(y_validation, dtype=torch.float32))
tensor_validation_dataloader = DataLoader(tensor_validation_ds, batch_size=batch_size, shuffle=False)

model.eval()
val_losses = []
with torch.no_grad():
    for features, targets in tensor_validation_dataloader:
        output = model(features).squeeze()
        loss = criterion(output, targets)
        val_losses.append(loss.item())

print(f"Validation Loss: {np.mean(val_losses)}")
