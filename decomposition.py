# 学习
# %%
from sklearn.decomposition import SparseCoder
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# 稀疏编码测试
def test_sparse_coder_estimator(X):
    n_components = 12
    n_features = 40
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    code = SparseCoder(dictionary=V, transform_algorithm='lasso_lars',
                       transform_alpha=0.001).transform(X)
    assert(not np.all(code == 0))
    assert(np.sqrt(np.sum((np.dot(code, V) - X) ** 2)) <= 0.1)

# %%
data = load_iris()
# type(data)
# data.keys()
# data['feature_names']
x = pd.DataFrame(data['data'])
y = data['target']

pca = PCA(n_components=2)
x_dr = pca.fit_transform(x)

# 可解释性方差/占比，大概就是重要性
pca.explained_variance_
pca.explained_variance_ratio_

# colors = ['red', 'blue', 'green']
# data['target_names']

# plt.figure()

# for i in range(3):
#     plt.scatter(x_dr[y == i, 0], \
#     x_dr[y == i, 1], \
#     alpha=0.7, \
#     c=colors[i], \
#     label=data['target_names'])

# plt.legend()
# plt.show()

# 展示随n增加，重要性的增加趋势
pca_line = PCA().fit(x)
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1,2,3,4]) #这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance")

# 按最大似然估计规则选择n
PCA(n_components='mle')
# 按信息比选择n
PCA(n_components=0.95)

