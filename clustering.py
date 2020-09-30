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

# %%
