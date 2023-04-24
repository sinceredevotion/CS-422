import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import KFold
from torchvision import transforms
import matplotlib.pyplot as plt

# Load the data
url = "http://mkang.faculty.unlv.edu/teaching/CS489_689/HW4/MNIST_HW4.csv"
data = pd.read_csv(url)

# Split into features and labels
X = data.iloc[:, 1:].values / 255.0  # normalize by dividing by 255
y = data.iloc[:, 0].values

# Convert to PyTorch tensors and reshape input images
X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 28, 28)
y = torch.tensor(y, dtype=torch.long)

# Define the CNN architecture
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.dropout = nn.Dropout(p=0.25)  # Dropout layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Second fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply first convolutional layer, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolutional layer, ReLU activation, and max pooling
        x = self.dropout(x)  # Apply dropout
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Apply first fully connected layer and ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply second fully connected layer
        return x

model = MNIST_CNN()

# Define the train function
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(train_loader), correct / total

# Define the evaluate function
def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(val_loader), correct / total

epochs = 20
batch_size = 32
kf = KFold(n_splits=5, shuffle=True, random_state=42)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
accuracy_list = []
all_train_accuracies = []
all_val_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Create data loaders
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Initialize the CNN model
    model = MNIST_CNN()

    # Set up the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    train_accuracies = []
    val_accuracies = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(f'Fold: {fold + 1}, Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    accuracy_list.append(val_accuracy)
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

# Calculate and print the average accuracy across all folds
average_accuracy = np.mean(accuracy_list)
print(f'Average 5-fold CV Accuracy: {average_accuracy:.4f}')

# Create a 2x3 grid of subplots for the learning curves
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Learning Curves and Accuracies for Each Fold')

# Plot the learning curves for each fold in separate subplots
for i, (train_accuracies, val_accuracies) in enumerate(zip(all_train_accuracies, all_val_accuracies)):
    ax = axes[i // 3, i % 3]
    ax.plot(train_accuracies, label=f'Training Accuracy (Fold {i + 1})')
    ax.plot(val_accuracies, label=f'Validation Accuracy (Fold {i + 1})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()

# Display the accuracies of each fold and their average as text
ax = axes[1, 2]
ax.axis('off')
y_pos = 0.8
for i, accuracy in enumerate(accuracy_list):
    ax.text(0.1, y_pos, f'Fold {i + 1}: {accuracy:.4f}', fontsize=12)
    y_pos -= 0.1

ax.text(0.1, y_pos, f'Average: {average_accuracy:.4f}', fontsize=12)

plt.show()