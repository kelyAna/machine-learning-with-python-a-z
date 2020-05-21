# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:24:43 2020

@author: lopes
"""


import pandas as pd

base = pd.read_csv('original.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

#LabelEnconder (categoricos -> numericos)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
                 
#importacao do naive baises
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

#gerando a tabela de probabilidade e ocorrendo a aprendizagem
classificador.fit(previsores,classe)

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)