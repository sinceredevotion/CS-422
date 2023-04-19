import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load training data
train_data = pd.read_csv('MNIST_training.csv')

# load test data
test_data = pd.read_csv('MNIST_test.csv')

# separate features (X) and labels (Y) for training data
TrainY = train_data.iloc[:, 0].values
TrainX = train_data.iloc[:, 1:].values

# separate features (X) and labels (Y) for test data
TestY = test_data.iloc[:, 0].values
TestX = test_data.iloc[:, 1:].values

# Normalize the training and test data using Z-score normalization
scaler = StandardScaler()
TrainX_norm = scaler.fit_transform(TrainX)
TestX_norm = scaler.transform(TestX)

# define function to calculate cosine similarity
def cosineSimilarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

# set the value of k
k = 15

# initialize empty list for predictions
predictions = []

# loop over each test sample
for o in range(len(TestX_norm)):
    distances = []

    # loop over each training sample to compute distance/similarity
    for i in range(len(TrainX_norm)):
        # calculate the cosine similarity between two samples
        distances.append(cosineSimilarity(TestX_norm[o], TrainX_norm[i]))

    # find the k nearest neighbors
    nearest_indices = np.argsort(distances)[-k:]
    nearest_labels = TrainY[nearest_indices]

    # decide the majority class among the k nearest neighbors
    prediction = np.bincount(nearest_labels).argmax()
    predictions.append(prediction)

# calculate the overall accuracy
correct = 0
incorrect = 0
for i in range(len(TestY)):
    if predictions[i] == TestY[i]:
        correct += 1
    else:
        incorrect += 1

accuracy = correct / len(TestY)

# print the accuracy and the number of correct and incorrect predictions
print("The amount of correct predictions is: ", correct)
print("The amount of incorrect predictions is: ", incorrect)
print("The accuracy is: ", accuracy)