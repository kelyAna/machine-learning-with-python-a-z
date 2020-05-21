import pandas as pd

df = pd.read_csv('house_prices.csv')

X = df.iloc[:, 3:19].values
y = df.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size = 0.3)

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes = (9,9))
regressor.fit(X_train, y_train)
score = regressor.score(X_test, y_test)

previsoes = regressor.predict(X_test)
y_test = scaler_y.inverse_transform(y_test)
previsoes = scaler_y.inverse_transform(previsoes)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, previsoes)


