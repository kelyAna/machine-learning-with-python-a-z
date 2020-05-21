# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:59:00 2020

@author: lopes
"""
import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')


#preencher os valores manualmente com a media
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

#divide a base em features e labels
features = base.iloc[:,1:4].values
labels = base.iloc[:,4].values

#verifica quais linhas tem a idade nula
base.loc[pd.isnull(base['age'])]

#substitui os campos nulos
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(features[:, 0:4])
features[:, 0:4] = imputer.transform(features[:,0:4])

#escalonamento de atributos, padronizar valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

#dividir a base em treino e teste
from sklearn.model_selection import train_test_split
features_treinamento, features_teste, labels_treinamento, labels_teste = train_test_split(features, labels, test_size=0.20, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
classificador.fit(features_treinamento, labels_treinamento)
previsoes = classificador.predict(features_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(labels_teste, previsoes)
matriz = confusion_matrix(labels_teste, previsoes)

