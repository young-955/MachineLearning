#学习
# %%
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import graphviz

w = load_wine()
# help(w)
w['feature_names']
# %%
# 决策树
# 参数选择 
# criterion: entropy, ginni
t = tree.DecisionTreeClassifier(criterion='entropy', random_state=50)

# 划分测试集
tr_d, ts_d, tr_t, ts_t = train_test_split(w.data, w.target, test_size=0.3)
t = t.fit(tr_d, tr_t)

# 计算分数
score = t.score(ts_d, ts_t)

# 获取叶子节点索引
t.apply([ts_d[0]])

# 可视化导出
fe_name = ['1','2','4','5','24','2413','类黄酮','非黄烷类酚类','花青素', 'a','q2', 'qv','bt']
dd = tree.export_graphviz(t, feature_names=fe_name, class_names=['的', 'VB薇恩', '帮我'], filled=True, rounded=True)
gr = graphviz.Source(dd)
gr

# 参数权重查看
[*zip(fe_name, t.feature_importances_)]


# 练习
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1)
plt.scatter(x[:, 0], x[:, 1])


rng = np.random.RandomState(1)
x += 2 * rng.uniform(size=x.shape)
plt.scatter(x[:, 0], x[:, 1])

dataset = [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=0), (x, y)]

fig = plt.figure(figsize=(6, 9))

i = 1

for ix, d in enumerate(dataset):
    x, y = d
    x = StandardScaler().fit_transform(x)

    tr_x, ts_x, tr_y, ts_y = train_test_split(x, y, test_size=0.3, random_state=1)

    x1_min, x1_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    x2_min, x2_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

    a1, a2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2), np.arange(x2_min, x2_max, 0.2))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#0000FF', '#FF0000'])

    ax = plt.subplot(len(dataset), 2, i)

    if ix == 0:
        ax.set_title('Input')
    
    ax.scatter(tr_x[:, 0], tr_x[:, 1], c=tr_y, cmap=cm_bright, edgecolors='k')
    ax.scatter(ts_x[:, 0], ts_x[:, 1], c=ts_y, cmap=cm_bright, alpha=0.6, edgecolors='k')

    ax.set_xlim(a1.min(), a1.max())
    ax.set_ylim(a2.min(), a2.max())
    ax.set_xticks(())
    ax.set_yticks(())

    i += 1

    ax = plt.subplot(len(dataset), 2, i)

    md = DecisionTreeClassifier(max_depth=5)
    md.fit(tr_x, tr_y)
    score = md.score(ts_x, ts_y)

    Z = md.predict_proba(np.c_[a1.ravel(), a2.ravel()])[:, 1]

    Z = Z.reshape(a1.shape)
    ax.contourf(a1, a2, Z, cmap=cm_bright, alpha=0.8)

    ax.scatter(tr_x[:, 0], tr_x[:, 1], c=tr_y, cmap=cm_bright, edgecolors='k')
    ax.scatter(ts_x[:, 0], ts_x[:, 1], c=ts_y, cmap=cm_bright, alpha=0.6, edgecolors='k')

    ax.set_xlim(a1.min(), a1.max())
    ax.set_ylim(a2.min(), a2.max())
    ax.set_xticks(())
    ax.set_yticks(())

    if ix == 0:
        ax.set_title('DT')

    ax.text(a1.max() - .3, a2.min() + .3, ('{:.1f}%'.format(score*100)), \
        size=15, horizontalalignment='right')

    i += 1
plt.show()
# %%
