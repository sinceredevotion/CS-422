import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable

# Custom linear regression implementation
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_features = X.shape[1]
        num_samples = X.shape[0]

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Perform gradient descent optimization
        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (2 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Load the dataset
file_path = "auto-mpg.data"
column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
df = pd.read_csv(file_path, delim_whitespace=True, names=column_names)

# Remove the 'car_name' column
df = df.drop(columns=["car_name"])

# Replace '?' with NaN and drop rows with missing values
df = df.replace("?", np.nan).dropna()

# Split the dataset into features (X) and target (y)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Implement 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize variables to store coefficients and RMSE values
coefficients = []
rmse_values = []

# Iterate through the 10 folds
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Normalize the training data
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # Train the custom linear regression model
    model = CustomLinearRegression()
    model.fit(X_train, y_train)

    # Normalize the test data using the mean and std from the training data
    X_test = scaler_x.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Calculate the coefficients and RMSE
    coefficients.append(model.weights)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)

# Create a PrettyTable to display the results
headers = ["Fold", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "RMSE"]
table = PrettyTable(field_names=headers)

for i, (coef, rmse) in enumerate(zip(coefficients, rmse_values)):
    row_data = [i + 1] + list(coef) + [rmse]
    table.add_row(row_data)

# Display the table
print("Results:")
print(table)

