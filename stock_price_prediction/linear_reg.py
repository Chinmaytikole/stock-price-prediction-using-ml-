import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
import plotly.graph_objects as go
import scipy.stats
import streamlit as st
from datetime import datetime
from stock_info import StockDataFetcher


class StockPricePredictor:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predicted = None

        # Fetch data and train the model upon initialization
        self.fetch_data()
        self.train_model()

    def fetch_data(self):
        # Download the stock data
        fetcher = StockDataFetcher(self.ticker, self.start_date, self.end_date)
        self.data = fetcher.fetch_data().copy()
        self.data.drop('Adj Close', axis=1, inplace=True)

        # Prepare features and target variable
        X = self.data.drop(['Close'], axis=1)
        y = self.data['Close']

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0,shuffle=False)

    def train_model(self):
        # Train the linear regression model
        regressor = LinearRegression()
        self.model = regressor.fit(self.X_train, self.y_train)

        # Make predictions
        self.predicted = regressor.predict(self.X_test)
        self.df = pd.DataFrame()
        self.df['close'] = self.y_test
        self.df['Predicted_close'] = self.predicted
        #print(self.df)

    def plot_actual_vs_predicted(self):
        # Plot actual vs predicted prices
        self.df['index']  = self.df.index
        plt.figure(figsize=(10, 5))
        plt.plot( self.df['close'], label='Actual Close Prices')
        plt.plot(self.df['index'], self.df['Predicted_close'], label='Predicted Close Prices', linestyle='--')
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted Stock Prices")
        plt.show()
        st.pyplot(plt)

    def plot_residuals(self):
        # Calculate and plot residuals
        plt.figure(figsize=(3,2))
        residual = self.y_test - self.predicted
        sns.displot(residual, kde=True)
        plt.title("Residuals Distribution")
        plt.show()
        st.pyplot(plt, use_container_width=False )
    def box_plot(self):
        f,axes = plt.subplots(1,4)
        sns.boxplot(y = "Open", data=self.data,ax = axes[0])
        sns.boxplot(y = "High", data=self.data,ax = axes[1])
        sns.boxplot(y = "Low", data=self.data,ax = axes[2])
        sns.boxplot(y = "Close", data=self.data,ax = axes[3])
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)
        plt.close()
    def candlestick_plot(self):
        self.data["date"]=self.data.index
        figure = go.Figure(data=[go.Candlestick(x=self.data["date"],
                                                open = self.data["Open"],
                                                high = self.data["High"],
                                                low=self.data["Low"],
                                                close = self.data["Close"])])
        figure.update_layout(title = f"{self.ticker} candlestick chart",
                             xaxis_rangeslider_visible=False)
        st.plotly_chart(figure, use_container_width=True)

    def model_summary(self):
        # Get OLS regression summary
        results = sm.OLS(self.y_test, self.X_test).fit()
        st.write(results.summary())

    def accuracy(self):
        x2 = abs(self.predicted - self.y_test)
        y2 = 100 * (x2 / self.y_test)
        accuracy = 100 - np.mean(y2)
        st.write(f"### Accuracy: {round(accuracy, 2)}%.")




    def evaluate_model(self):
        # Evaluate model performance
        mae = metrics.mean_absolute_error(self.y_test, self.predicted)
        mse = metrics.mean_squared_error(self.y_test, self.predicted)
        rmse = np.sqrt(mse)
        p_value = scipy.stats.norm.sf(abs(1.67))

        st.write("Mean Absolute Error: ", mae)
        st.write("Mean Squared Error: ", mse)
        st.write("Root Mean Squared Error: ", rmse)
        st.write('p-value: ', p_value)

    def predicting(self,user_date):
        # Input for user date
        #user_date = st.text_input("Enter the next Date for which you want to predict (YYYY-MM-DD):")

        # Check if the user has entered a date
        if user_date:
            try:
                # Convert the user input into a datetime object
                user_date_datetime = datetime.strptime(user_date, '%Y-%m-%d')

                # Fetch the last available data row for feature values
                last_row = self.data.iloc[-1]
                last_features = last_row[['Open', 'High', 'Low', 'Volume']].values.reshape(1,-1)  # Reshape for prediction

                # Add a placeholder for the future date (here we assume the date does not affect these features)
                future_open = last_row['Open']  # Use the last known Open
                future_high = last_row['High']  # Use the last known High
                future_low = last_row['Low']  # Use the last known Low
                future_volume = last_row['Volume']  # Use the last known Volume

                # Create an input array for prediction
                future_input = np.array([[future_open, future_high, future_low, future_volume]])

                # Use the model to predict the closing price for the specified date
                y = self.model.predict(future_input)

                # Display the prediction
                st.subheader(f"{y[0]:.2f}")

            except ValueError:
                st.error("Please enter a valid date in the format YYYY-MM-DD.")

# Example usage:
# predictor = StockPricePredictor('^NSEI', '2020-07-01', '2024-09-30')
# predictor.plot_actual_vs_predicted()
# predictor.plot_residuals()
# predictor.evaluate_model()
# predictor.model_summary()