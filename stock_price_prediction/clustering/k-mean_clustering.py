import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from stock_info import StockDataFetcher

# Fetch the stock data
fetcher = StockDataFetcher("^NSEI", '2024-4-15', '2024-10-15')
data = fetcher.fetch_data()

# Convert 'Close' column into a 2D NumPy array for K-Means
X = data['Close'].values.reshape(-1, 1)

# Apply K-Means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot the clusters
plt.scatter(X, np.zeros_like(X), c=y_kmeans, s=50, cmap='viridis')
plt.xlabel('Close Price')
plt.ylabel('Cluster Label')
plt.title('K-Means Clustering of Close Prices')
plt.show()

# Plot the centroids
centers = kmeans.cluster_centers_
plt.scatter(centers, np.zeros_like(centers), c='red', s=200, alpha=0.75, marker='x')
plt.title('Centroids of the Clusters')
plt.show()
