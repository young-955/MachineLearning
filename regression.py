#
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 关闭科学计数法显示
np.set_printoptions(suppress=True)

data = load_breast_cancer()

x = data['data']
y = data['target']
data['target_names']
data['feature_names']

lrl1 = LR(penalty='l1', solver='liblinear', C=0.5, max_iter=1000)
lrl2 = LR(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)

lrl1 = lrl1.fit(x, y)
lrl2 = lrl2.fit(x, y)

# print(lrl1.coef_)
# print(lrl2.coef_)

l1 = []
l2 = []
l1_test = []
l2_test = []
tr_x, ts_x, tr_y, ts_y = train_test_split(x, y, test_size=0.3, random_state=2)
for i in np.linspace(0.05, 5, 100):
    lrl1 = LR(penalty='l1', solver='liblinear', C=i, max_iter=1000)
    lrl2 = LR(penalty='l2', solver='liblinear', C=i, max_iter=1000)

    lrl1.fit(tr_x, tr_y)
    lrl2.fit(tr_x, tr_y)

    l1.append(accuracy_score(tr_y, lrl1.predict(tr_x)))
    l2.append(accuracy_score(tr_y, lrl2.predict(tr_x)))
    l1_test.append(accuracy_score(ts_y, lrl1.predict(ts_x)))
    l2_test.append(accuracy_score(ts_y, lrl2.predict(ts_x)))

graph = [l1, l2, l1_test, l2_test]
color = ['red','blue', 'gray','black']
label = ['l1','l2', 'l1test','l2test']
plt.figure(figsize=(6,6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05, 5, 100), graph[i], color[i], label=label[i])
plt.legend(loc=4)
plt.show()

# %%
