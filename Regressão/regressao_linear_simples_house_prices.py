# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 01:10:26 2020

@author: lopes
"""
import pandas as pd

df = pd.read_csv('house_prices.csv')

X = df.iloc[:,5:6].values
Y = df.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,
                                                    test_size = 0.3,
                                                    random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
score = regressor.score(x_train,y_train)

import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
plt.plot(x_train, regressor.predict(x_train), color = 'red')
plt.show()

previsoes = regressor.predict(x_test)

resultado = abs(y_test - previsoes)
resultado.mean()

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)

plt.scatter(x_test,y_test)
plt.plot(x_test, regressor.predict(x_test), color = 'red')
plt.show()

regressor.score(x_test, y_test)
