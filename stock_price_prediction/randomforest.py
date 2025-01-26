import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from stock_info import StockDataFetcher
import streamlit as st


class StockAnalysisrf:
    def __init__(self, stock_symbol, start_date, end_date):
        # Initialize with stock symbol and fetch data
        self.stock_symbol = stock_symbol
        # Automatically perform analysis upon initialization
        self.stock_data = self.fetching_data(start_date, end_date)
        self.add_indicators()
        self.classify_data()
        self.prepare_data()
        try:
            self.train_random_forest()
        except Exception as e:
            st.write(f"No sufficient data: {e}")

    def fetching_data(self, start_date=None, end_date=None):
        # Fetch historical stock data from Yahoo Finance
        fetcher = StockDataFetcher(self.stock_symbol, start_date, end_date)
        stock_data = fetcher.fetch_data()
        return stock_data

    def add_indicators(self):
        # Adding technical indicators: EMA and RSI
        self.stock_data['LEMA'] = ta.ema(self.stock_data['Close'], length=10)
        self.stock_data['RSI'] = ta.rsi(self.stock_data['Close'], length=14)
        self.stock_data.dropna(inplace=True)

    def classify_data(self):
        # Create classification based on EMA and RSI
        self.stock_data['ema_class'] = np.where(self.stock_data['LEMA'] > self.stock_data['Close'], 1, 0)

        rsi_midpoint = (self.stock_data['RSI'].max() + self.stock_data['RSI'].min()) / 2
        self.stock_data['rsi_class'] = np.where(self.stock_data['RSI'] > rsi_midpoint, 1, 0)

        # Shift the 'Close' column to create the 'next_cls' column
        self.stock_data['next_cls'] = self.stock_data['Close'].shift(-1)
        self.stock_data['cls_class'] = np.where(self.stock_data['Close'] < self.stock_data['next_cls'], 1, 0)
        st.write(self.stock_data)

    def prepare_data(self):
        # Prepare features and target
        self.stck_data_class = self.stock_data[['Low', 'High', 'Close', 'ema_class', 'rsi_class', 'cls_class']].copy()

    def train_random_forest(self):
        # Features and target for model training
        X = self.stck_data_class.drop(['Low', 'High', 'Close', 'cls_class'], axis=1)  # Features
        y = self.stck_data_class['cls_class']  # Target

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        # Create and train the RandomForestClassifier
        self.rf_model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=1)
        self.rf_model.fit(X_train, y_train)

        # Make predictions and calculate accuracy
        y_pred = self.rf_model.predict(X_test)
        self.y_test = y_test
        self.y_pred = y_pred
        self.accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write(f"Random Forest Model Accuracy: {self.accuracy:.2f}")

    def plot_data(self):
        try:
            # Plot stock data along with technical indicators
            plt.figure(figsize=(15, 8))
            self.stock_data['Close'].div(self.stock_data['Close'][0]).mul(100).plot(label='Close Price')
            self.stock_data['LEMA'].div(self.stock_data['LEMA'][0]).mul(100).plot(label='LEMA')
            self.stock_data['RSI'].div(self.stock_data['RSI'][0]).mul(100).plot(label='RSI')
            plt.title(f"{self.stock_symbol} Stock Data with EMA & RSI")
            plt.legend()
            plt.show()
            st.pyplot(plt)
            plt.close()
        except:
            None

    def plot_actual_vs_predicted_scatter(self):
        try:
            # Scatter plot of actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(self.y_test)), self.y_test, label="Actual", color='blue', marker='o', s=50)
            plt.scatter(range(len(self.y_pred)), self.y_pred, label="Predicted", color='orange', marker='x', s=50)

            plt.xlabel("Time (Index)")
            plt.ylabel("Class (Up/Down)")
            plt.title("Actual vs Predicted Stock Movement (Scatter Plot)")
            plt.legend()
            plt.show()
            st.pyplot(plt)
            plt.close()
        except AttributeError:
            st.write("Model not trained or no predictions available")

    def get_accuracy(self):
        # Return model accuracy
        try:
            return self.accuracy
        except:
            None
