import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


df = pd.read_csv('combined_data.csv')
print("Dataset size:", df.shape)
print("Dataset head \n", df.head())

# Drop non-numeric columns
df_numeric = df.drop(columns=['category', 'subcategory', 'is_new', 'brand', 'codCountry'])

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)  # Example with 3 clusters
kmeans.fit(df_numeric)

# Add cluster labels to the dataframe
df_numeric['cluster'] = kmeans.labels_

# View the results
print(df_numeric.head())

# Initialize the KMeans object with a chosen number of clusters
kmeans = KMeans(n_clusters=3, random_state=42)  # Using 3 clusters as an example

# Fit the KMeans model
kmeans.fit(df_numeric)
kmeans.fit(df_numeric)
kmeans.fit(df_numeric)
kmeans.fit(df_numeric)

# Add the cluster labels to the DataFrame
df_numeric['cluster'] = kmeans.labels_

# View the first few rows to see the cluster assignments
print(df_numeric.head())


# Calculate the mean values of features for each cluster
cluster_means = df_numeric.groupby('cluster').mean()

# Display the mean values for each cluster
print(cluster_means)

# Plot the clusters based on different feature pairs
plt.figure(figsize=(12, 6))

# Example scatter plot for 'current_price' vs 'likes_count'
plt.scatter(df_numeric['discount'], df_numeric['likes_count'], c=df_numeric['cluster'])
plt.xlabel('discount')
plt.ylabel('Likes Count')
plt.title('Cluster Visualization: discount vs Likes Count')
plt.show()
