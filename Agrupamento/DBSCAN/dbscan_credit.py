import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('credit_card_clients.csv', header = 1)
df['BILL_TOTAL'] = df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']

X = df.iloc[:,[1,25]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

dbscan = DBSCAN(eps = 0.40, min_samples = 4)
previsoes = dbscan.fit_predict(X)
unicos, qtd = np.unique(previsoes, return_counts = True)

plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s = 100, c = 'red', label = 'Clusters 1')
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s = 100, c = 'yellow', label = 'Clusters 2')
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], s = 100, c = 'green', label = 'Clusters 3')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()


lista_clientes = np.column_stack((df, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]
