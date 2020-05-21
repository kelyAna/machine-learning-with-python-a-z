# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:46:31 2020

@author: lopes
"""

import pandas as pd
import numpy as np

df = pd.read_csv('house_prices.csv')

X = df.iloc[:, 3:19].values
y = df.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X,y,
                                                    random_state = 0)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
X_treinamento_poly = poly.fit_transform(X_treinamento)
X_teste_poly = poly.transform(X_teste)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento_poly, y_treinamento)
score = regressor.score(X_teste_poly, y_teste)

previsoes = regressor.predict(X_teste_poly)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)
