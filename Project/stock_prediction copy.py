import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from pmdarima import auto_arima



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: default, 1: INFO, 2: WARNING, 3: ERROR

def load_data(path, min_data_points=500):
    all_files = os.listdir(path)
    all_data = []

    for file in all_files:
        if file.endswith('.txt'):
            filepath = os.path.join(path, file)
            try:
                data = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=True)
                if len(data) < min_data_points:
                    print(f"Skipping file {file} due to insufficient data points.")
                    continue
                all_data.append(data)
            except pd.errors.EmptyDataError:
                print(f"Skipping file {file} due to empty data.")
                continue

    combined_data = pd.concat(all_data, axis=0, ignore_index=False)
    return combined_data

def preprocess_data(data):
    data = data[['Close']].resample('D').mean().dropna()
    data = data.asfreq('D', method='pad')

    data = data.sort_index()  # Ensure the date index is sorted and monotonic
    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    return train_data, test_data

def train_sarima(train_data, sarima_order, seasonal_order):
    model = SARIMAX(train_data, order=sarima_order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit

def train_lstm(train_data, n_input, n_features, lstm_units, epochs, batch_size):
    generator = TimeseriesGenerator(train_data.values, train_data.values, length=n_input, batch_size=batch_size)

    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(generator, epochs=epochs)
    return model

def train_exponential_smoothing(train_data, trend=None, seasonal=None, smoothing_level=None, smoothing_slope=None, smoothing_seasonal=None):
    model = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal)
    model_fit = model.fit(smoothing_level=smoothing_level, smoothing_slope=smoothing_slope, smoothing_seasonal=smoothing_seasonal)
    return model_fit

def evaluate_sarima(model, test_data):
    predictions = model.forecast(steps=len(test_data))
    mse = mean_squared_error(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)
    return predictions, mse, mae, r2

def evaluate_lstm(model, n_input, test_data):
    test_generator = TimeseriesGenerator(test_data.values, test_data.values, length=n_input, batch_size=1)
    predictions = model.predict(test_generator)

    mse = mean_squared_error(test_data.values[n_input:], predictions)
    mae = mean_absolute_error(test_data.values[n_input:], predictions)
    r2 = r2_score(test_data.values[n_input:], predictions)
    return predictions, mse, mae, r2

def evaluate_exponential_smoothing(model, test_data):
    predictions = model.forecast(steps=len(test_data))
    mse = mean_squared_error(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)
    return predictions, mse, mae, r2

def find_best_sarima_params(train_data, seasonal=True, seasonal_periods=365, max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2):
    model = auto_arima(train_data, seasonal=seasonal, m=seasonal_periods, max_p=max_p, max_d=max_d, max_q=max_q, max_P=max_P, max_D=max_D, max_Q=max_Q, suppress_warnings=True, stepwise=False, trace=True, n_jobs=8)
    return model.order, model.seasonal_order

def plot_results(test_data, sarima_preds, lstm_preds, exp_smoothing_preds):
    plt.figure(figsize=(14, 8))
    
    plt.plot(test_data.index, test_data.values, label='Actual', color='blue', linewidth=1)
    plt.plot(test_data.index, sarima_preds, label='SARIMA', color='red', linewidth=1)
    plt.plot(test_data.index[n_input:], lstm_preds, label='LSTM', color='green', linewidth=1)
    plt.plot(test_data.index, exp_smoothing_preds, label='Exponential Smoothing', color='orange', linewidth=1)
    
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.show()

path = "Project/archive/Data/Stocks"
all_data = load_data(path)
train_data, test_data = preprocess_data(all_data)

n_input = 60
n_features = 1
lstm_units = 50
epochs = 10
batch_size = 32
trend = 'additive'
seasonal = 'multiplicative'
smoothing_level = 0.6
smoothing_slope = 0.2
smoothing_seasonal = 0.2

sarima_order, seasonal_order = find_best_sarima_params(train_data)


sarima_model = train_sarima(train_data, sarima_order, seasonal_order)
lstm_model = train_lstm(train_data, n_input, n_features, lstm_units, epochs, batch_size)
exp_smoothing_model = train_exponential_smoothing(train_data, trend=trend, seasonal=seasonal, smoothing_level=smoothing_level, smoothing_slope=smoothing_slope, smoothing_seasonal=smoothing_seasonal)

sarima_preds, sarima_mse, sarima_mae, sarima_r2 = evaluate_sarima(sarima_model, test_data)
lstm_preds, lstm_mse, lstm_mae, lstm_r2 = evaluate_lstm(lstm_model, n_input, test_data)
exp_smoothing_preds, exp_smoothing_mse, exp_smoothing_mae, exp_smoothing_r2 = evaluate_exponential_smoothing(exp_smoothing_model, test_data)

plot_results(test_data, sarima_preds, lstm_preds, exp_smoothing_preds)
