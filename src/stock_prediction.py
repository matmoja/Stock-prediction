from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class StockPrediction:
    def __init__(self, api_key):
        self.api_key = api_key
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')

    def get_stock_data(self, symbol, interval='1min', output_size='compact'):
        try:
            data, meta_data = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize=output_size)
            return data
        except ValueError as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return None
        
    def get_historical_data(self, symbol, interval='daily', output_size='full'):
        try:
            data, meta_data = self.ts.get_daily(symbol=symbol, outputsize=output_size)
            return data
        except ValueError as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None
        
    def engineer_features(self, data, close_column='4. close'):
        """
        Engineer features for stock prediction.

        Parameters:
        - data: DataFrame containing historical stock data.

        Returns:
        - DataFrame with engineered features.
        """
            
        data.loc[:, 'daily_return'] = data[close_column].pct_change()
       
        data = data.dropna()

        return data
    
    def split_data(self, data, test_size=0.2):
        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

        return train_data, test_data
    
    def train_model(self, train_data):
        features = train_data[['daily_return']]
        target = train_data['4. close']

        scaler = StandardScaler()
        features_standardized = scaler.fit_transform(features)

        model = LinearRegression()
        model.fit(features_standardized, target)

        return model
    
    def evaluate_model(self, model, test_data):
        features_test = test_data[['daily_return']]
        target_test = test_data['4. close']

        scaler = StandardScaler()
        features_test_standardized = scaler.fit_transform(features_test)

        predictions = model.predict(features_test_standardized)

        mse = mean_squared_error(target_test, predictions)
        mae = mean_absolute_error(target_test, predictions)
        r2 = r2_score(target_test, predictions)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared: {r2}")
        
    def make_predictions(self, model, new_data):
        if new_data.empty:
            return []
        
        engineered_data = self.engineer_features(new_data)

        features_new_data = engineered_data[['daily_return']]

        scaler = StandardScaler()
        features_new_data_standardized = scaler.fit_transform(features_new_data)

        predictions = model.predict(features_new_data_standardized)

        return predictions






if __name__ == "__main__":
    # Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
    api_key = 'YOUR_API_KEY'
    stock_prediction = StockPrediction(api_key)

    symbol = 'AAPL'

    # Fetching stock data for the last 5 days with 1-minute interval
    stock_data = stock_prediction.get_stock_data(symbol, interval='1min', output_size='compact')
    
    # Splitting data into training and testing sets
    train_data, test_data = stock_prediction.split_data(stock_data)
    
    print("Training Data:")
    print(train_data)
    print("\nTesting Data:")
    print(test_data)
