import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the housing data
data = pd.read_csv('housing_training.csv', header=None)

# Rename the columns 10, 12, and 13 to K, M, and N respectively
data = data.rename(columns={10: 'K', 12: 'M', 13: 'N'})

# Create a boxplot of the columns K, M, and N
data.boxplot(column=['K', 'M', 'N'])

# Add a title to the plot
plt.title("Box Plot of Columns K, M, and N in Housing Data")

# Add labels to the x and y axis
plt.xlabel("Column")
plt.ylabel("Value")

# Display the plot
plt.show()