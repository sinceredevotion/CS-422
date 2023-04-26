import os
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from prophet import Prophet
from tqdm import tqdm

tensorflow.keras.utils.disable_interactive_logging()


def load_data(path):
    files = os.listdir(path)
    stock_data = []

    for file in tqdm(files, desc="Loading stock data", position=0, leave=True):
        if file.endswith('.txt'):
            file_path = os.path.join(path, file)
            if os.stat(file_path).st_size > 0:  # Check if file is not empty
                stock = pd.read_csv(file_path, sep=',', header=0)
                
                # Check for the existence of the 'date' column and rename it if necessary
                if 'date' not in stock.columns:
                    # Check if there's an alternative name for the date column (e.g., 'Date')
                    if 'Date' in stock.columns:
                        stock.rename(columns={'Date': 'date'}, inplace=True)
                    else:
                        print(f"Skipping file {file} due to missing 'date' column.")
                        continue
                
                stock_data.append(stock)

    return pd.concat(stock_data, axis=0, ignore_index=True)

def preprocess_data(data):
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # Drop duplicate index values
    data = data.loc[~data.index.duplicated(keep='first')]
    
    data = data.asfreq('B')
    data.fillna(method='ffill', inplace=True)

    return data

def train_test_split(data, ratio):
    train_size = int(len(data) * ratio)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    return train, test

def arima_model(train, order):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    return model_fit


def lstm_model(train, look_back, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    generator = TimeseriesGenerator(train['Close'].values, train['Close'].values, length=look_back, batch_size=batch_size)
    model.fit(generator, epochs=epochs, verbose=0)

    return model

def prophet_model(train):
    train = train.reset_index()[['date', 'Close']]  # Keep only 'date' and 'Close' columns
    train.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    print("Training Prophet model...")
    model.fit(train)
    print("Prophet model training completed.")
    return model

def predict_arima(model, test):
    forecast = model.forecast(steps=len(test))
    return forecast

def predict_lstm(model, train, test, look_back):
    predictions = []
    input_data = train['Close'][-look_back:].values

    for _ in range(len(test)):
        input_data = input_data.reshape((1, look_back, 1))
        prediction = model.predict(input_data)
        predictions.append(prediction[0][0])
        input_data = np.append(input_data[:, 1:], prediction)

    return np.array(predictions)


def predict_prophet(model, test):
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-len(test):].values

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}


def visualize_results(test, arima_pred, lstm_pred, prophet_pred):
    print("Generating plot...")
    
    plt.figure(figsize=(14, 7))
    plt.plot(test.index, test.values, label='True Prices', color='blue')
    plt.plot(test.index, arima_pred, label='ARIMA Predictions', color='green')
    plt.plot(test.index, lstm_pred, label='LSTM Predictions', color='red')
    plt.plot(test.index, prophet_pred, label='Prophet Predictions', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

def main():
    path = 'Project/archive/Data/Stocks'
    data = load_data(path)
    data = preprocess_data(data)

    # Filter Apple stock data
    apple_stock_data = data[data['file_name'] == 'aapl.us.txt']
    train, test = train_test_split(apple_stock_data, ratio=0.8)

    # Use only 'Close' prices for ARIMA model
    arima_train = train['Close']
    arima_test = test['Close']

    # ARIMA
    arima_order = (5, 1, 0)
    arima = arima_model(arima_train, arima_order)
    arima_pred = predict_arima(arima, arima_test)

    # LSTM
    look_back = 10
    epochs = 50
    batch_size = 32
    lstm = lstm_model(train, look_back, epochs, batch_size)
    lstm_pred = predict_lstm(lstm, train, test, look_back)

    # Prophet
    prophet = prophet_model(train)
    prophet_pred = predict_prophet(prophet, test)

    # Evaluate Models
    arima_metrics = evaluate_model(test['Close'].values, arima_pred)
    lstm_metrics = evaluate_model(test['Close'].values, lstm_pred)
    prophet_metrics = evaluate_model(test['Close'].values, prophet_pred)

    print("ARIMA metrics (MSE, MAE, R2):", arima_metrics)
    print("LSTM metrics (MSE, MAE, R2):", lstm_metrics)
    print("Prophet metrics (MSE, MAE, R2):", prophet_metrics)

    # Visualize Results
    visualize_results(test['Close'], arima_pred, lstm_pred, prophet_pred)

if __name__ == '__main__':
    main()
