import numpy as np
from sklearn.cluster import KMeans


def initial_seeding(data, train_size, batch_size, X_train):

    if data > 2:
        X = []
        for i in range(train_size):
            X.append(X_train[i].flatten())
        X = np.array(X)
    else:
        X = X_train

    kmeans = KMeans(n_clusters=batch_size, init='k-means++')
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    centers = []
    for i in range(centroids.shape[0]):
        # 簇中的点
        cluster_points = X[labels == i]
        cluster_points_idx = np.arange(train_size)[labels == i]

        centroid = centroids[i]  # 质心
        distances = np.linalg.norm(cluster_points - centroid, axis=1)  # 计算距离
        centers.append(cluster_points_idx[np.argmin(distances)])  # 最近的点

    return centers
