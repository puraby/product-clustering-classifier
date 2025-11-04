Load retail datasets
Pulls 9 CSVs (accessories, bags, etc.) from GitHub, keeps columns like category, brand, current_price, discount, likes_count, merges them into one DataFrame, fills missing brand with “Unknown”.

K-Means clustering (unsupervised)
Uses only 4 numeric features—current_price, raw_price, discount, likes_count—to cluster items into 3 groups.
→ Saves the cluster id to a new column label.

Cluster profiling
Calculates average price/discount/likes per cluster and how many items are in each; also shows which categories dominate each cluster.

Hierarchical clustering (Ward)
Takes a 1,000-row sample of those 4 numeric features, standardizes them, builds a dendrogram, cuts at a distance threshold to assign another cluster label on the sample—mainly for visualization/validation of cluster structure.

Supervised models to predict the K-Means label
Builds a new table with features (category, subcategory, prices, discount, is_new, brand) and targets the K-Means label.
Splits into train/test, then trains:

a Decision Tree, does a small GridSearch;

a KNN classifier, also with GridSearch;
and prints accuracies + confusion matrices.
