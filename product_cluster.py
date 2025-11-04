import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# -------------------- TASK  1 ---------------------------------------------

PATH = 'https://raw.githubusercontent.com/masudf6/bigdata_analytics/main/A1_2024_Released/'

def import_csvs(csv_files, selected_columns):
    dataframes = [pd.read_csv(PATH+csv_file, usecols=selected_columns) for csv_file in csv_files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

csv_files = [
    'accessories.csv', 'bags.csv', 'beauty.csv', 'house.csv',
    'jewelry.csv', 'kids.csv', 'men.csv', 'shoes.csv', 'women.csv'
]

selected_columns = [
    'category', 'subcategory', 'current_price', 'raw_price',
    'discount', 'likes_count', 'is_new', 'brand'
]

combined_df = import_csvs(csv_files, selected_columns)

# Check for null values in each column
null_values = combined_df.isnull().sum()
print(null_values[null_values > 0])

# Visualize null values
# sns.heatmap(combined_df.isnull(), cbar=False, cmap='viridis')
# plt.title('Null Values Heatmap')
# plt.show()

# Replace the NULL values with 'unknown'
combined_df['brand'].fillna('Unknown', inplace=True)

# Check the dataset after handling Nulls
# sns.heatmap(combined_df.isnull(), cbar=False, cmap='viridis')
# plt.title('Null Values Heatmap')
# plt.show()

print('Task 1')
print(combined_df.head())



from sklearn.cluster import KMeans

# Selecting relevant numerical features for KMeans clustering
features = ['current_price', 'raw_price', 'discount', 'likes_count']

# inertia = []

# # Testing different numbers of clusters
# k_values = range(1, 11)  # Testing from 1 to 10 clusters
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(combined_df[features])
#     inertia.append(kmeans.inertia_)

# # Plotting the elbow curve
# plt.figure(figsize=(8, 5))
# plt.plot(k_values, inertia, marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Sum of Square')
# plt.title('Elbow Method for Optimal Number of Clusters')
# plt.xticks(k_values)
# plt.grid(True)
# plt.show()

# --------------------- KMeans clustering -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
combined_df['label'] = kmeans.fit_predict(combined_df[features])

print('Task 2')
print(combined_df.head())

cluster_means = combined_df.groupby('label')[['current_price', 'raw_price', 'discount', 'likes_count']].mean()
cluster_sizes = combined_df['label'].value_counts().sort_index()
cluster_details = cluster_means.copy()
cluster_details['Cluster Size'] = cluster_sizes

print(cluster_details)

# 3D scatter plot: current_price, discount, likes_count, colored by cluster
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# sc = ax.scatter(combined_df['current_price'], combined_df['discount'], combined_df['likes_count'], c=combined_df['label'], cmap='viridis', s=50, alpha=0.7)
# ax.set_title('Current Price vs Discount vs Likes Count')
# ax.set_xlabel('Current Price')
# ax.set_ylabel('Discount')
# ax.set_zlabel('Likes Count')

# plt.colorbar(sc, label='Cluster')
# plt.show()

# Create subplots grid with 2 rows and 3 columns
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
columns = features

# Initialize subplot indices
row, col = 0, 0

# Iterate through pairs of columns
# for i in range(len(columns)-1):
#     for j in range(i+1, len(columns)):
#         axs[row, col].scatter(combined_df[columns[i]], combined_df[columns[j]], c=combined_df['label'])
#         axs[row, col].set_title(f'{columns[i]} vs {columns[j]}')

#         # Update subplot indices
#         col += 1
#         if col >= 3:  # Move to the next row after three columns
#             col = 0
#             row += 1

# plt.tight_layout()
# plt.show()

# Filtering the DataFrame for Cluster 4
cluster_4_data = combined_df[combined_df['label'] == 1]

# Counting the occurrences of each category in Cluster 4
most_frequent_category_cluster_4 = cluster_4_data['category'].value_counts()
print(most_frequent_category_cluster_4)

# Grouping by 'label' (cluster) and 'category', and then counting the occurrences
category_cluster_frequency = combined_df.groupby(['category', 'label']).size().unstack(fill_value=0)

# Displaying the frequency of each category in all clusters
print(category_cluster_frequency)


# -------------- Hierarchical Clustering ------------------
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

data = combined_df

numerical_features = data[['current_price', 'raw_price', 'discount', 'likes_count']]

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

# # Scatter plot of likes_count vs. Discount, highlighting Cluster 6
# plt.figure(figsize=(10, 6))
# plt.scatter(sampled_data['discount'], sampled_data['likes_count'], c=sampled_data['Cluster'], cmap='viridis', label='Clusters')
# plt.colorbar(label='Cluster')
# plt.xlabel('likes_count')
# plt.ylabel('Discount')
# plt.title('likes_count vs. Discount by Cluster')
# plt.scatter(cluster_means_sampled.loc[6, 'likes_count'], cluster_means_sampled.loc[6, 'discount'], color='red', s=200, edgecolor='black', label='Cluster 6')
# plt.legend()
# plt.show()


# -------------------- Task 3 ------------------------------------------------
# df = combined_df.copy()
df = combined_df[['category', 'subcategory', 'current_price', 'raw_price', 'discount', 'is_new', 'brand', 'label']].copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le_features = ['category', 'subcategory', 'brand', 'is_new']

for feature in le_features:
    encoded_values = le.fit_transform(df[feature][df[feature] != 'unknown'])
    df.loc[df[feature] != 'unknown', feature] = encoded_values

df = df.replace('unknown', -29)

print(df.head())

# Test Train
X = df.drop('label', axis=1)
y = df['label']

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Fit scaler on training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------- Decision Tree ------------------------------
from sklearn.tree import DecisionTreeClassifier

clf_default = DecisionTreeClassifier(random_state=42)
clf_default.fit(X_train, y_train)

# Visualize the tree structure. Just show the first four layers
from sklearn import tree

# fig, ax = plt.subplots(figsize=(20, 20))
# tree.plot_tree(clf_default, max_depth=4, filled=True, fontsize=10)
# plt.show()

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluate the trained model with the testing data
y_pred = clf_default.predict(X_test)
# The prediction accuracy
accuracy = accuracy_score(y_pred, y_test)
print('The testing accuracy is: %.4f\n' % accuracy)

# Show the confusion matrix
labels = clf_default.classes_
cm = confusion_matrix(y_pred, y_test)
print('Confusion Matrix')
print(cm)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],            # Criteria to measure the quality of a split
    'splitter': ['best', 'random'],              # Strategy to split at each node
    'max_depth': [None, 10, 20, 30, 40, 50],     # Maximum depth of the tree
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=clf_default, param_grid=param_grid, scoring='accuracy', cv=5)
# grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and their corresponding performance
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate the model with the best hyperparameters on the test set
best_rf = grid_search.best_estimator_
test_score = best_rf.score(X_test, y_test)
print("Test Set Score:", test_score)

# Visualize the tree structure. Just show the first four layers
# from sklearn import tree

# fig, ax = plt.subplots(figsize=(20, 20))
# tree.plot_tree(best_rf, max_depth=4, filled=True, fontsize=10)
# plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
# plt.show()

#------------------- KNN --------------------------------------
# Train KNN model for classification
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Evaluate the trained model with the testing data
y_pred = knn.predict(X_test)
# The prediction accuracy
accuracy = accuracy_score(y_pred, y_test)
print('The testing accuracy of KNN is: %.4f\n' % accuracy)

# Confusion matrix
cm = confusion_matrix(y_pred, y_test)
print('KNN confusion matrix', cm)

# Parameter Grid for KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],            # Number of neighbors to consider
    'weights': ['uniform', 'distance'],          # Weight function used in prediction
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='accuracy', cv=5)
# grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and their corresponding performance
print("Best Hyperparameters for KNN:", grid_search.best_params_)
print("Best Score for KNN:", grid_search.best_score_)

# Evaluate the model with the best hyperparameters on the test set
best_rf = grid_search.best_estimator_
test_score = best_rf.score(X_test, y_test)
print("Final test score KNN:", test_score)