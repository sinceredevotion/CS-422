import pandas as pd
import matplotlib.pyplot as plt

# Read the housing data from a csv file
data = pd.read_csv('housing_training.csv', header=None)

# Plot a histogram of column 0 (A)
data[0].hist()

# Add labels to x and y axis
plt.xlabel("Column A")
plt.ylabel("Frequency")

# Add a title to the graph
plt.title("Histogram of Column A in Housing Data")

# Display the histogram
plt.show()