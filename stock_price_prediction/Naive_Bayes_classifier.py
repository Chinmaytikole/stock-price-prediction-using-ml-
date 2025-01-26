from stock_info import StockDataFetcher
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


class NaiveBayesClassifier:
    def __init__(self, stock_symbol,start_date,end_date):
        # Fetch stock data
        fetcher = StockDataFetcher(stock_symbol,start_date,end_date)
        self.stock_data = fetcher.fetch_data().copy()

        # Adding technical indicators: Exponential Moving Average (EMA) and Relative Strength Index (RSI)
        self.stock_data['LEMA'] = ta.ema(self.stock_data['Close'], length=10)
        self.stock_data['RSI'] = ta.rsi(self.stock_data['Close'], length=14)

        # Dropping any rows with NaN values
        self.stock_data = self.stock_data.dropna()

        # Create classification based on the EMA indicator
        self.stock_data['ema_class'] = np.where(self.stock_data['LEMA'] > self.stock_data['Close'], 1, 0)

        # Compute the mid-point of the RSI range for classification
        rsi_midpoint = (self.stock_data['RSI'].max() + self.stock_data['RSI'].min()) / 2
        self.stock_data['rsi_class'] = np.where(self.stock_data['RSI'] > rsi_midpoint, 1, 0)

        # Shift the 'Close' column to create the 'next_cls' column
        self.stock_data['next_cls'] = self.stock_data['Close'].shift(-1)

        # Classify based on the next day's price movement
        self.stock_data['cls_class'] = np.where(self.stock_data['Close'] < self.stock_data['next_cls'], 1, 0)

        # Prepare features and target
        self.X = self.stock_data[['ema_class', 'rsi_class']].copy()  # Features
        self.y = self.stock_data['cls_class']  # Target

        # Split data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=1)

        # Initialize and train the Naive Bayes model
        self.nb_model = GaussianNB()
        self.nb_model.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.nb_model.predict(self.X_test)

        # Evaluate the model
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        self.conf_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)
        self.class_report = metrics.classification_report(self.y_test, self.y_pred)

    def get_accuracy(self):
        return self.accuracy

    def display_confusion_matrix(self):
        plt.figure(figsize=(8, 6))
        metrics.ConfusionMatrixDisplay(self.conf_matrix).plot()
        plt.title("Confusion Matrix")
        plt.show()
        st.pyplot(plt)
        plt.close()

    def display_classification_report(self):
        st.write(f" Classification Report:\n{self.class_report} ")


