import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from stock_info import StockDataFetcher
import pandas_ta as ta
import numpy as np

class StockAssociationRules:
    def __init__(self, stock_symbol, start_date, end_date):
        # Fetch stock data
        fetcher = StockDataFetcher(stock_symbol, start_date, end_date)
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

        # Prepare boolean transaction DataFrame
        self.transactions = self.stock_data[['ema_class', 'rsi_class']].copy()

        # Convert to boolean type for better performance with Apriori
        self.transactions = self.transactions.astype(bool)

        # Display the transactions dataset
        print("Transactions Data:")
        print(self.transactions)

        # Apply the Apriori algorithm to find frequent itemsets with a minimum support of 50%
        self.frequent_itemsets = apriori(self.transactions, min_support=0.5, use_colnames=True)

        # Generate association rules from the frequent itemsets based on confidence
        self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=0.7)

        # Display results
        self.display_results()

    def display_results(self):
        print("\nFrequent Itemsets:")
        print(self.frequent_itemsets)

        print("\nAssociation Rules:")
        print(self.rules)

# Example usage
if __name__ == "__main__":
    stock_symbol = "^NSEI"  # Replace with the desired stock symbol
    start_date = '2024-01-01'  # Start date for fetching data
    end_date = '2024-10-15'  # End date for fetching data

    stock_association = StockAssociationRules(stock_symbol, start_date, end_date)
