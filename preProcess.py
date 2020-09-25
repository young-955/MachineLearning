# 学习
# %%
# 0,1的缩放
from sklearn.preprocessing import MinMaxScaler
# 正态分布的缩放, x - x.min / D(x)
from sklearn.preprocessing import StandardScaler

# 以minmaxscaler为例，standardscaler类似
data = [[1,3,4],[12,25,24],[43,35,51],[16,8,22]]
scaler = MinMaxScaler((-1, 1))
res = scaler.fit_transform(data)

# 特征多时fit会报错，此时使用以下接口
scaler.partial_fit(data)


# %%
# 类别标签转数值
from sklearn.preprocessing import LabelEncoder
# 分类特征转数值
from sklearn.preprocessing import OrdinalEncoder
# onehot
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
# 类别标签
l = ['a','b','c','a']
l = pd.DataFrame(l)
le = LabelEncoder()
le.fit_transform(l)
# 获取类别
le.classes_

# 分类特征
data = [[1,3,4],[12,25,24],[43,35,51],[16,8,22]]
o = OrdinalEncoder()
o.fit_transform(pd.DataFrame(data))
# 获取类别
o.categories_


# %%
# 二值化
from sklearn.preprocessing import Binarizer

a = [[1,23,4,6,213]]
b = Binarizer(5)
b.fit_transform(a)

# %%
