from stock_info import StockDataFetcher
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
import numpy as np
from sklearn import metrics

class StockPredictor:
    def __init__(self, stock_symbol,start_date,end_date):
        # Fetch stock data during initialization
        self.fetcher = StockDataFetcher(stock_symbol,start_date,end_date)
        self.data = self.fetcher.fetch_data()

        # Prepare the data
        self.data = self.data.reset_index()  # Reset index to access 'Date' as a column
        self.data['Date_ordinal'] = pd.to_datetime(self.data['Date']).map(pd.Timestamp.toordinal)

        # X will be the date (as ordinal numbers), y will be the closing price
        self.X = self.data[['Date_ordinal']]
        self.y = self.data['Close']

        # Initialize and fit the Linear Regression model
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

        # Predict the closing prices for the existing data
        self.data['Predicted_Close'] = self.model.predict(self.X)
        print(self.data['Predicted_Close'])

        # Predict the closing price for the next day
        last_date = self.data['Date'].max()
        next_day_ordinal = pd.to_datetime(last_date).toordinal() + 1
        self.next_day_prediction = self.model.predict([[next_day_ordinal]])
        self.next_day_date = datetime.fromordinal(next_day_ordinal)
    def accuracy(self):
        x2 = abs(self.data['Predicted_Close'] - self.y)
        y2 = 100 * (x2 / self.y)
        accuracy = 100 - np.mean(y2)
        st.write(f"### Accuracy: {round(accuracy, 2)}%.")

    def predicting(self):
        # Input for user date
        user_date = st.text_input("Enter the next Date for which you want to predict (YYYY-MM-DD):")

        # Check if the user has entered a date
        if user_date:
            try:
                # Convert the user input into a datetime object
                user_date_datetime = datetime.strptime(user_date, '%Y-%m-%d')

                # Convert the date to an ordinal number for prediction
                user_date_ordinal = user_date_datetime.toordinal()

                # Use the model to predict the closing price for the specified date
                y = self.model.predict([[user_date_ordinal]])

                # Display the prediction
                st.header(f"On the Date: {user_date} Closing price will be: {y[0]:.2f}")


            except ValueError:
                st.error("Please enter a valid date in the format YYYY-MM-DD.")

    def get_coefficients(self):
        """Returns the coefficient and intercept of the trained model."""
        return self.model.coef_[0], self.model.intercept_
    def predictingf(self,user_date):
        st.subheader(self.next_day_prediction)

    def plot_results(self):
        """Plots the actual and predicted prices along with next day's prediction."""
        last_date = self.data['Date'].max()

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['Date'], self.data['Close'], label='Actual Close Prices')
        plt.plot(self.data['Date'], self.data['Predicted_Close'], label='Predicted Close Prices', linestyle='--')
        plt.axvline(x=last_date, color='red', linestyle=':', label='Last Date')
        plt.scatter(self.next_day_date, self.next_day_prediction, color='green', label='Next Day Prediction', zorder=5)
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Linear Regression on Stock Prices (with Next Day Prediction)')
        plt.legend()
        plt.show()
        st.pyplot(plt)
        plt.close

# Usage in another file
# from stock_predictor import StockPredictor

# Initialize the StockPredictor class with a stock symbol
predictor = StockPredictor('HAVELLS.NS','2024-09-22','2024-10-22')

# Get model coefficients
coef, intercept = predictor.get_coefficients()
print(f"Coefficient: {coef}, Intercept: {intercept}")

# Plot the results
predictor.plot_results()

# Get model coefficients

