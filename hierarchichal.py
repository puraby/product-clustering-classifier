import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('combined_data.csv')

# Selecting numerical features for clustering
numerical_features = data[['current_price', 'raw_price', 'discount', 'likes_count']]

# Sampling a subset of the data to avoid memory issues
sampled_data = numerical_features.sample(n=1000, random_state=42)

# Standardizing the sample data
scaler = StandardScaler()
scaled_sampled_data = scaler.fit_transform(sampled_data)

# Performing hierarchical clustering using the 'ward' method on the sampled data
linked_sampled = linkage(scaled_sampled_data, method='ward')

# Plotting the dendrogram for the sampled data
plt.figure(figsize=(10, 7))
dendrogram(linked_sampled, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Sampled Data)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Determining the optimal number of clusters by applying a threshold
max_d = 15  # Example threshold value
clusters_sampled = fcluster(linked_sampled, max_d, criterion='distance')


# Assigning clusters back to the sampled data
sampled_data['Cluster'] = clusters_sampled
print(sampled_data)

# Analyzing the characteristics of each cluster
cluster_means_sampled = sampled_data.groupby('Cluster').mean()

print("Cluster Means:")
print(cluster_means_sampled)

# Visualizing the clusters
# Scatter plot of Current Price vs. Likes Count, highlighting Cluster 6
plt.figure(figsize=(10, 6))
plt.scatter(sampled_data['current_price'], sampled_data['likes_count'], c=sampled_data['Cluster'], cmap='viridis', label='Clusters')
plt.colorbar(label='Cluster')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.title('Current Price vs. Likes Count by Cluster')
plt.scatter(cluster_means_sampled.loc[6, 'current_price'], cluster_means_sampled.loc[6, 'likes_count'], color='red', s=200, edgecolor='black', label='Cluster 6')
plt.legend()
plt.show()

# Scatter plot of Raw Price vs. Discount, highlighting Cluster 6
plt.figure(figsize=(10, 6))
plt.scatter(sampled_data['raw_price'], sampled_data['discount'], c=sampled_data['Cluster'], cmap='viridis', label='Clusters')
plt.colorbar(label='Cluster')
plt.xlabel('Raw Price')
plt.ylabel('Discount')
plt.title('Raw Price vs. Discount by Cluster')
plt.scatter(cluster_means_sampled.loc[6, 'raw_price'], cluster_means_sampled.loc[6, 'discount'], color='red', s=200, edgecolor='black', label='Cluster 6')
plt.legend()
plt.show()

# Scatter plot of likes_count vs. Discount, highlighting Cluster 6
plt.figure(figsize=(10, 6))
plt.scatter(sampled_data['likes_count'], sampled_data['discount'], c=sampled_data['Cluster'], cmap='viridis', label='Clusters')
plt.colorbar(label='Cluster')
plt.xlabel('likes_count')
plt.ylabel('Discount')
plt.title('likes_count vs. Discount by Cluster')
plt.scatter(cluster_means_sampled.loc[6, 'likes_count'], cluster_means_sampled.loc[6, 'discount'], color='red', s=200, edgecolor='black', label='Cluster 6')
plt.legend()
plt.show()