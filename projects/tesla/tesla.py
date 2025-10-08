# -*- coding: utf-8 -*-

# -- Sheet --

import pandas                  as     pd
import matplotlib.pyplot       as     plt
import seaborn                 as     sns
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing   import StandardScaler 
from   sklearn.linear_model    import LinearRegression
from   sklearn.metrics         import mean_absolute_error
from   sklearn.tree            import DecisionTreeRegressor
from   sklearn.ensemble        import RandomForestRegressor
from   sklearn.svm             import SVR

# # **Prédire le prix de l'action Tesla**


df = pd.read_csv('tesla_2014_2023.csv')

df.head()

df.isnull().sum()

df.shape

df.info()

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.head()

# ## **Comment se comporte l'action Tesla ?**


plt.figure(figsize=(20, 15))
plt.plot(df.index, df['open'], label='Open')
plt.plot(df.index, df['close'], label='Close')
plt.plot(df.index, df['high'], label='High')
plt.plot(df.index, df['low'], label='Low')
plt.legend()

# ## **Quelle est la corélation de chaque colonne ?**
# `next_day_close` est la colonne cible.


df_corr = df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(df_corr, annot=True, fmt='.1f')

# Son score est 100 %  avec `open`, `close`et `low`.  
# Cest trois colonnes influences `next_day_close`.


# ### **Diviser l'ensemble en features et lavels**


y = df['next_day_close']
X = df.drop(columns='next_day_close')

# ### **Un ensemble pour l'entraînement, l'autre pour le test**


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### **Normaliser l'esemble avant l'entraînement**


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_train_scaled.shape, X_test_scaled.shape

# ### **Régression linéaire**


dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
dtr_prediction = dtr.predict(X_test)
dtr_mae = mean_absolute_error(y_test, dtr_prediction)
dtr_mae

# $ 3,94  
# Ce n'est pas bien.


# ### **Régression Random Forest**


rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)
rfr_mae = mean_absolute_error(y_test, rfr_prediction)
rfr_mae

# $ 2,95  
# Mieux.


# ### **Régresseur Vecteur de support**


svr = SVR()
svr.fit(X_train, y_train)
svr_prediction = svr.predict(X_test)
svr_mae = mean_absolute_error(y_test, svr_prediction)
svr_mae

# $ 73  
# C'est très mauvais.
# 
# Le modèle de régression Random Forest est mieux.


