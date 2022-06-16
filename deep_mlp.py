# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 01:40:07 2022

@author: Klay
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd

# SB_UPJONG_CD = 업종코드 / SEX_CCD = 성별 / USECT_CORR = 결제횟수
# M = 1 / F = 0
df = pd.read_csv("dataset/processed_data.csv")
df = df.values

data = np.zeros((len(df[:]), 2))
data[:, 0] = df[:, 0]
data[:, 1] = df[:, 2]

x_train, x_test, y_train, y_test = train_test_split(data, df[:, 1], train_size=0.8)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32, 64, 128),
                    activation="relu",
                    learning_rate_init=0.001,
                    batch_size=32,
                    solver="sgd",
                    verbose=True)

mlp.fit(x_train, y_train)
print(mlp.score(x_test, y_test))

mglearn.plots.plot_2d_separator(mlp, x_train, fill=True, alpha=.3)
mglearn.discrete_scatter(x_train[:, 0], x_train[:, 1], y_train)
plt.xlabel("Store Code")
plt.ylabel("Count of Receipt")

# plt.plot(mlp.loss_curve_)