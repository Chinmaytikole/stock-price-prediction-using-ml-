import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from stock_info import StockDataFetcher

# Fetch the stock data
fetcher = StockDataFetcher("^NSEI", '2024-4-15', '2024-10-15')
data = fetcher.fetch_data()

# Convert 'Close' column into a 2D NumPy array for Agglomerative Clustering
X = data['Close'].values.reshape(-1, 1)

# Apply Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=4)
y_agg = agg_cluster.fit_predict(X)

# Plot the clusters
plt.scatter(X, np.zeros_like(X), c=y_agg, s=50, cmap='viridis')
plt.xlabel('Close Price')
plt.ylabel('Cluster Label')
plt.title('Agglomerative Clustering of Close Prices')
plt.show()
