# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 00:21:40 2020

@author: lopes
"""

import pandas as pd

df = pd.read_csv('plano_saude.csv')

X = df.iloc[:,0].values
Y = df.iloc[:,1].values

import numpy as np
correlacao = np.corrcoef(X,Y)

X = X.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,Y)

#b0
regressor.intercept_

#b1
regressor.coef_

import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title("Regressão Linear Simples")
plt.xlabel('Idade')
plt.ylabel('Custo')
plt.show()

#previsão pessoa com 40 anos
previsao1 = regressor.intercept_ + regressor.coef_ * 40
previsao2 = regressor.predict(np.array(40).reshape(1, -1))

score = regressor.score(X,Y)

from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X,Y)
visualizador.poof()
plt.show()


