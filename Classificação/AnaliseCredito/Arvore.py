# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import pandas as pd
import numpy as np

base = pd.read_csv('diabetes.csv')



#divide a base em features e labels
features = base.iloc[:,1:8].values
labels = base.iloc[:,8].values


#substitui os campos nulos
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(features[:, 0:8])
features[:, 0:8] = imputer.transform(features[:,0:8])

#escalonamento de atributos, padronizar valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

#dividir a base em treino e teste
from sklearn.model_selection import train_test_split
features_treinamento, features_teste, labels_treinamento, labels_teste = train_test_split(features, labels, test_size=0.20, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy', random_state=None)
classificador.fit(features_treinamento, labels_treinamento)
previsoes = classificador.predict(features_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(labels_teste, previsoes)
matriz = confusion_matrix(labels_teste, previsoes)
