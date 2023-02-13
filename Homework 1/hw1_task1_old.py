import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import the necessary libraries for data processing and visualization

# Read MNIST data from a csv file
data = pd.read_csv('MNIST_100.csv')

# Create two variables:
# y: the target variable, in this case the labels for each digit (0-9)
# X: the feature set, in this case the pixel values for each image
y = data.iloc[:, 0] # select the first column as target variable
X = data.drop('label', axis=1) # drop the 'label' column and assign the rest to the feature set

# Apply PCA to reduce the dimensions of the feature set
pca = PCA(n_components=2) # create an instance of PCA with  components
pca.fit(X) # fit the PCA model to the feature set
PCAX = pca.transform(X) # transform the feature set to the PCA components

# Visualize the transformed data
plt.scatter(PCAX[0:100, 0], PCAX[0:100, 1]) # Digit 0
plt.scatter(PCAX[100:200, 0], PCAX[100:200, 1]) # Digit 1
plt.scatter(PCAX[200:300, 0], PCAX[200:300, 1]) # Digit 2
plt.scatter(PCAX[300:400, 0], PCAX[300:400, 1]) # Digit 3
plt.scatter(PCAX[400:500, 0], PCAX[400:500, 1]) # Digit 4
plt.scatter(PCAX[500:600, 0], PCAX[500:600, 1]) # Digit 5
plt.scatter(PCAX[600:700, 0], PCAX[600:700, 1]) # Digit 6
plt.scatter(PCAX[700:800, 0], PCAX[700:800, 1]) # Digit 7
plt.scatter(PCAX[800:900, 0], PCAX[800:900, 1]) # Digit 8
plt.scatter(PCAX[900:1000, 0], PCAX[900:1000, 1]) # Digit 9

# Add legend
labels = np.unique(y) # get the unique labels in the target variable
plt.legend(labels) # add the legend with the unique labels

# Show the plot
plt.show()

# Return the PCA transformed data
PCAX
