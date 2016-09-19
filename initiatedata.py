#!/usr/bin/env python
# coding:utf-8
import numpy as np
import chainer.functions as F
from chainer import Variable, FunctionSet, optimizers

#モデルを作る
model = F.Linear(3,3)

#これがモデルに与えるベクトル
data = np.array([[1,2,3]] , dtype=np.float32)

x = Variable(np.array(data))

#ここでモデルにベクトルを投げて、加工させている
y = model(x)

#モデルが加工したベクトルを表示
print(y.data) 