#
# %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

x, y = make_blobs(500, 2, 4, random_state=3)

# fig, ax1 = plt.subplots(1)

# ax1.scatter(x[:, 0], x[:, 1], s=8)

# color = ['red', 'green', 'yellow', 'blue']
# fig, ax1 = plt.subplots(1)
# for i in range(4):
#     ax1.scatter(x[y==i, 0], x[y==i, 1], c=color[i], s=8)
# plt.show()

from sklearn.cluster import KMeans

k = KMeans(n_clusters=3, random_state=3)
l = k.fit(x)
l.labels_
l.cluster_centers_
l.inertia_
# help(l)

pre = l.fit_predict(x)

# color = ['red', 'green', 'yellow', 'blue']
# fig, ax1 = plt.subplots(1)
# for i in range(4):
#     ax1.scatter(x[pre==i, 0], x[pre==i, 1], c=color[i], s=8)
# plt.show()

# 计算轮廓系数
from sklearn.metrics import silhouette_samples, silhouette_score

silhouette_score(x, pre)
silhouette_score(x, l.labels_)
silhouette_samples(x, pre)


#%%
n_clusters = 4
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
ax1.set_xlim([-0.1, 1])

ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])
clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
cluster_labels = clusterer.labels_
silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)
sample_silhouette_values = silhouette_samples(X, cluster_labels)

y_lower = 10

for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i)/n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper)
                    ,ith_cluster_silhouette_values
                    ,facecolor=color
                    ,alpha=0.7)
    ax1.text(-0.05
             , y_lower + 0.5 * size_cluster_i
             , str(i))
    y_lower = y_upper + 10

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([])
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

ax2.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', alpha=1, s=200)
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")
plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')
plt.show()

