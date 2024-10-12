import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
file_path = '/content/archive (1).zip'
customers_df = pd.read_csv(file_path)
X = customers_df[['Annual Income (k$)', 'Spending Score (1-100)']]
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
kmeans = KMeans(n_clusters=5, random_state=42)
customers_df['Cluster'] = kmeans.fit_predict(X)
print("Cluster centers:\n", kmeans.cluster_centers_)
plt.figure(figsize=(10, 6))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=customers_df['Cluster'], cmap='rainbow', s=100, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='X', label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()