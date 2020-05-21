# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:15:02 2020

@author: lopes
"""

#visualização de dados com pandas
import pandas as pd
base = pd.read_csv('census.csv')

#descrição de dados
base.describe()

#divisão da base em atributos e classe
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

#importacao da biblioteca sklearn, para transformar atributos nominais em numéricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#selecionando todas as colunas que serão transformadas
labelencoder_previsores = LabelEncoder()
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)