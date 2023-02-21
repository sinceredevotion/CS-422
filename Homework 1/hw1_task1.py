import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Read MNIST data from a csv file
data = pd.read_csv('MNIST_100.csv')

# Create two variables:
# y: the target variable, in this case the labels for each digit (0-9)
# X: the feature set, in this case the pixel values for each image
y = data.iloc[:, 0] # select the first column as target variable
X = data.drop('label', axis=1) # drop the 'label' column and assign the rest to the feature set

# Apply PCA to reduce the dimensions of the feature set
pca = PCA(n_components=2) # create an instance of PCA with 2 components
pca.fit(X) # fit the PCA model to the feature set
PCAX = pca.transform(X) # transform the feature set to the PCA components


# Visualize the transformed data with different colors for each group
for i in range(10):
    group = PCAX[i*100:(i+1)*100, :]
    plt.scatter(group[:, 0], group[:, 1], label='Group {}'.format(i))

# Add labels and title
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('MNIST Digits Visualization using PCA')

# Add legend
plt.legend(title='Digits')

# Show the plot
plt.show()