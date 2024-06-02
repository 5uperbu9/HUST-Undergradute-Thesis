import numpy as np
from sklearn.cluster import KMeans


def initial_seeding(train_size, batch_size, X):

    kmeans = KMeans(n_clusters=batch_size, init='k-means++')
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    centers = []

    for i in range(centroids.shape[0]):
        cluster_points = X[labels == i]
        cluster_points_idx = np.arange(train_size)[labels == i]
        centroid = centroids[i]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_point_idx = np.argmin(distances)
        centers.append(cluster_points_idx[closest_point_idx])

    return centers
