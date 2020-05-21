# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:02:51 2020

@author: lopes
"""

import pandas as pd
import numpy as np

df = pd.read_csv('plano_saude.csv')

X = df.iloc[:, 0:1].values
y = df.iloc[:,1].values

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X,y)
score1 = regressor1.score(X,y)

regressor1.predict(np.array(40).reshape(1, -1))

import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.plot(X, regressor1.predict(X), color = 'red')
plt.title('Regressão Linear')
plt.xlabel('Idade')
plt.ylabel('Custo')


#regressao polinomial

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly,y)
score2 = regressor2.score(X_poly,y)

regressor2.predict(poly.transform(np.array(40).reshape(1, -1)))

plt.scatter(X,y)
plt.plot(X, regressor2.predict(X_poly), color = 'yellow')
plt.title('Regressão Linear Polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')





