import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pmdarima
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from fbprophet import Prophet

def load_data(file_path):
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    stock_data = data[['Close']]
    return stock_data

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

def main():
    # Load the dataset
    file_path = "path/to/stock_data.csv"
    stock_data = load_data(file_path)

    # ARIMA Model
    from pmdarima import auto_arima

    # Split the data into train and test sets
    train_data, test_data = stock_data[:int(len(stock_data)*0.8)], stock_data[int(len(stock_data)*0.8):]

    # Fit the ARIMA model
    model_arima = auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True, trace=True)
    model_arima.fit(train_data)

    # Make predictions
    predictions_arima = model_arima.predict(n_periods=len(test_data))
    predictions_arima = pd.DataFrame(predictions_arima, index=test_data.index, columns=['Prediction'])

    # Calculate metrics
    mse_arima = mean_squared_error(test_data, predictions_arima)
    mae_arima = mean_absolute_error(test_data, predictions_arima)
    r2_arima = r2_score(test_data, predictions_arima)

    # Plot actual and predicted prices
    plt.plot(train_data, label='Train')
    plt.plot(test_data, label='Actual')
    plt.plot(predictions_arima, label='ARIMA Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # LSTM Model
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.optimizers import Adam

    # Data normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data)

    # Create time series dataset
    def create_dataset(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size - 1):
            X.append(data[i:(i + window_size), 0])
            y.append(data[i + window_size, 0])
        return np.array(X), np.array(y)

    window_size = 10
    X, y = create_dataset(stock_data_scaled, window_size)

    # Train-test split
    train_size = int(len(stock_data_scaled) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Build LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, input_shape=(1, window_size)))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

    # Train the model
    model_lstm.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    # Make predictions
    predictions_lstm = model_lstm.predict(X_test)
    predictions_lstm = scaler.inverse_transform(predictions_lstm)

    # Calculate metrics
    mse_lstm = mean_squared_error(y_test, predictions_lstm)
    mae_lstm = mean_absolute_error(y_test, predictions_lstm)
    r2_lstm = r2_score(y_test, predictions_lstm)

    # Plot actual and predicted prices
    plt.plot(y_test, label='Actual')
    plt.plot(predictions_lstm, label='LSTM Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


    # Prophet Model
    import fbprophet
    from fbprophet import Prophet

    # Prepare the dataset for Prophet
    stock_data_prophet = stock_data.reset_index().rename(columns={"Date": "ds", "Close": "y"})

    # Train-test split
    train_data_prophet, test_data_prophet = stock_data_prophet[:int(len(stock_data_prophet)*0.8)], stock_data_prophet[int(len(stock_data_prophet)*0.8):]

    # Fit the Prophet model
    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(train_data_prophet)

    # Make predictions
    future = model_prophet.make_future_dataframe(periods=len(test_data_prophet))
    predictions_prophet = model_prophet.predict(future)

    # Calculate metrics
    mse_prophet = mean_squared_error(test_data_prophet['y'], predictions_prophet['yhat'].iloc[-len(test_data_prophet):])
    mae_prophet = mean_absolute_error(test_data_prophet['y'], predictions_prophet['yhat'].iloc[-len(test_data_prophet):])
    r2_prophet = r2_score(test_data_prophet['y'], predictions_prophet['yhat'].iloc[-len(test_data_prophet):])

    # Plot actual and predicted prices
    fig, ax = plt.subplots()
    model_prophet.plot(predictions_prophet, ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    plt.legend(['Actual', 'Prophet Prediction'])
    plt.show()

if __name__ == '__main__':
    main()
