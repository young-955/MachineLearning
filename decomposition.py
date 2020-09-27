# 学习
# %%
from sklearn.decomposition import SparseCoder
import numpy as np
from sklearn.decomposition import PCA

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
#coding=utf-8

__author__ = 'Administrator'

import pythoncom
from win32com.shell import shell, shellcon

g_desk = None

def toGBK(s):
    return s.decode('utf-8').encode('gb2312')

def getDeskComObject():
    global g_desk
    if not g_desk:
        g_desk = pythoncom.CoCreateInstance(shell.CLSID_ActiveDesktop, \
                                             None, pythoncom.CLSCTX_INPROC_SERVER, \
                                             shell.IID_IActiveDesktop)
    return g_desk

def setWallPaper(paper):
    desktop = getDeskComObject()
    if desktop:
        desktop.SetWallpaper(paper, 0)
        desktop.ApplyChanges(shellcon.AD_APPLY_ALL)

def addUrlLink(lnk):
    desktop = getDeskComObject()
    desktop.AddUrl(0, lnk, 0, 0)

if __name__ == '__main__':
    paper = r'G:\meinv\长腿美女刘奕宁Lynn唯美私房照\16.jpg'
    setWallPaper(paper)
# %%
