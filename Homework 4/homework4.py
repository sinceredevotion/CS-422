import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable

# Load the dataset
df = pd.read_csv('MNIST_HW4.csv')

# Split the dataset into features and labels
X = df.drop('label', axis=1)
y = df['label']

# Define the kernels to be compared
kernels = ['linear', 'poly', 'rbf']

# Initialize the table with the PrettyTable library to display results
table = PrettyTable()
table.field_names = ["Kernel", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average"]

# Apply 5-fold cross-validation for each kernel
for kernel in kernels:
    # Create an SVM model with the current kernel
    svm_model = SVC(kernel=kernel)
    
    # Calculate accuracy scores using 5-fold cross-validation
    scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')
    
    # Calculate the average accuracy score for the current kernel
    avg_score = np.mean(scores)
    
    # Add the results to the table
    table.add_row([kernel.upper()] + [f'{score:.4f}' for score in scores] + [f'{avg_score:.4f}'])

# Print the table with the results
print(table)
