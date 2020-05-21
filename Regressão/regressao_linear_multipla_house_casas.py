# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 01:31:52 2020

@author: lopes
"""
import pandas as pd

df = pd.read_csv('house_prices.csv')

X = df.iloc[:,3:19].values
Y = df.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,
                                                    test_size = 0.3,
                                                    random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
score = regressor.score(x_train,y_train)

previsoes = regressor.predict(x_test)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test,previsoes)

regressor.score(x_test, y_test)
regressor.intercept_
len(regressor.coef_)