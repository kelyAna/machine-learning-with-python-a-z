# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:50:57 2020

@author: lopes
"""

import pandas as pd 

df = pd.read_csv('plano_saude2.csv')

X = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X,y)
score = regressor.score(X,y)


import numpy as np
X_teste = np.arange(min(X), max(X), 0.1)
X_teste = X_teste.reshape(-1,1)

import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.plot(X_teste,regressor.predict(X_teste), color = 'red')
plt.title('Regressão com árvores de decisão')
plt.xlabel('Idade')
plt.ylabel('Custo')
