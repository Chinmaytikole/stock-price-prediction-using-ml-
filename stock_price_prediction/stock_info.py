import yfinance as yf


class StockDataFetcher:
    def __init__(self, stock_symbol,start_date,end_date):
        self.stock_symbol = stock_symbol
        self.end_date = end_date
        self.start_date = start_date

    def fetch_data(self):
        stock_data = yf.download(self.stock_symbol, self.start_date,self.end_date)
        return stock_data


# fetcher = StockDataFetcher("^NSEI",'12-10-2022','12-10-2024')
#
# # Fetch the stock data
# data = fetcher.fetch_data()
#
# # Display the stock data
# print(data)
