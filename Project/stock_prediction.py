import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: default, 1: INFO, 2: WARNING, 3: ERROR

def load_data(path, min_data_points=500):
    all_files = os.listdir(path)
    stock_data = {}

    for file in all_files:
        if file.endswith('.txt'):
            filepath = os.path.join(path, file)
            try:
                data = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=True)
                if len(data) < min_data_points:
                    print(f"Skipping file {file} due to insufficient data points.")
                    continue
                stock_data[file[:-4]] = data
            except pd.errors.EmptyDataError:
                print(f"Skipping file {file} due to empty data.")
                continue

    stock_data = dict(sorted(stock_data.items(), key=lambda item: -len(item[1]))[:6])
    return stock_data

def preprocess_data(data):
    data = data[['Close']].resample('D').mean().dropna()
    data = data.asfreq('D', method='pad')

    data = data.sort_index()  # Ensure the date index is sorted and monotonic
    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    return train_data, test_data

def train_lstm(train_data, n_input, n_features, lstm_units, epochs, batch_size):
    generator = TimeseriesGenerator(train_data.values, train_data.values, length=n_input, batch_size=batch_size)

    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(generator, epochs=epochs)
    return model

def evaluate_lstm(model, n_input, test_data):
    test_generator = TimeseriesGenerator(test_data.values, test_data.values, length=n_input, batch_size=1)
    predictions = model.predict(test_generator)

    mse = mean_squared_error(test_data.values[n_input:], predictions)
    mae = mean_absolute_error(test_data.values[n_input:], predictions)
    r2 = r2_score(test_data.values[n_input:], predictions)
    return predictions, mse, mae, r2

def plot_results(stock_data, test_data, lstm_preds):
    n_stocks = len(stock_data)
    fig, axs = plt.subplots(n_stocks, 1, figsize=(14, 4 * n_stocks), sharex=True)

    for i, stock in enumerate(stock_data.keys()):
        axs[i].plot(test_data[stock].index, test_data[stock].values, label='Actual', color='blue', linewidth=1)
        axs[i].plot(test_data[stock].index[n_input:], lstm_preds[stock], label='LSTM', color='green', linewidth=1)
        
        axs[i].set_title(f'{stock} Stock Price Prediction')
        axs[i].legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

path = "Project/archive/Data/Stocks"
all_data = load_data(path)

n_input = 60
n_features = 1
lstm_units = 50
epochs = 30
batch_size = 32

train_data = {}
test_data = {}
lstm_models = {}
lstm_predictions = {}

for stock, data in all_data.items():
    train, test = preprocess_data(data)
    train_data[stock] = train
    test_data[stock] = test
    lstm_models[stock] = train_lstm(train, n_input, n_features, lstm_units, epochs, batch_size)
    lstm_predictions[stock], _, _, _ = evaluate_lstm(lstm_models[stock], n_input, test)

plot_results(all_data, test_data, lstm_predictions)