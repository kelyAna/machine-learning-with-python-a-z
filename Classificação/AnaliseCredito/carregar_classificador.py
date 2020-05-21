# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:38:35 2020

@author: lopes
"""


import pandas as pd
import numpy as np
import pickle

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

svm = pickle.load(open('svm_finalizado.sav','rb'))
random_forest = pickle.load(open('random_finalizado.sav', 'rb'))
mlp = pickle.load(open('mlp_finalizado.sav', 'rb'))

resultado_svm = svm.score(previsores, classe)
resultado_random = random_forest.score(previsores, classe)
resultado_mlp = mlp.score(previsores, classe)

novo_registro = [[50000, 40, 5000]]
novo_registro = np.asarray(novo_registro)
novo_registro = novo_registro.reshape(-1,1)
novo_registro = scaler.fit_transform(novo_registro)
novo_registro = novo_registro.reshape(-1,3)

resposta_svm = svm.predict(novo_registro)
resposta_random = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)

