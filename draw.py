# 可视化工具Yellowbrick

# plt
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def himmelblau(x):
	return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
X, Y = np.meshgrid(x, y)
Z = X + Y

fig = plt.figure(figsize=(12, 10))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)   # 画曲面
# ax.plot(X, Y, Z)       # 画曲线，好像x， y得是一维的
ax.view_init(60, -30)       # 好像是调成图的角度
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

# 3d绘图,plot、bar等类似
ax = plt.axes(projection='3d')
ax.scatter3D([1,2,3], [3,3,3], [3,6,9])
ax.bar3d([1,2,3], [3,4,5], [0,0,0], [.1, .1, .1], [.1, .1, .1], [3, 6, 9])
plt.show()

# 并列柱状图
w = 0.8
a = np.random.randint(5, 5)
b = np.random.randint(5, 5)
c = np.random.randint(5, 5)
plt.bar(x, a, width=w, label='a')
plt.bar(x + w, b, width=w, label='b')
plt.bar(x + 2 * w, c, width=w, label='c')
plt.legend()