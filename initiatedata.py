# データを生成してプロットする
from numpy.random import *
from sklearn import *
from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
np.random.seed(0)
X, y = skl.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)