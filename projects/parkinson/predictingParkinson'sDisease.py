# -*- coding: utf-8 -*-

# -- Sheet --

import pandas                  as pd
from   sklearn.preprocessing   import StandardScaler
from   sklearn.model_selection import train_test_split
from   sklearn.svm             import SVC
from   sklearn.metrics         import accuracy_score

df = pd.read_csv("parkinson_disease.csv")
df.shape

# Voir toutes les colonnes
pd.set_option('display.max_columns', 25)

df.info()

df.isnull().sum()

# Il n'y a pas de valeur manquante.  
# Que des nombres sauf name.  
# Name n'a pas d'intérêt pour ce projet.  
# Pas de valeur catégorielle.


df.drop(columns='name', inplace=True)

# Diviser en feature et label
y = df['status']
X = df.drop(columns='status')

# Quelles sont les valeur des colonnes de fetures
X['NHR'].min(), X['NHR'].max()
X['MDVP:Flo(Hz)'].min(), X['MDVP:Flo(Hz)'].max()

# Aucune variables de features n'est normalisées.  
# Il est impossible d'alimenter un algo de ML.
# 
# Normaliser avec une échelle strandard est une option.


scaler = StandardScaler()
X = scaler.fit_transform(X)
X

# Après la normalisation de df devient un tableau NumPy.  
# Le type array NumPy n'est pas un souci pour alimenter le modèle de ML.
# 
# Je vais tout de même le convertir en dataframe.


X_columns = df.drop(columns='status').columns
pd.DataFrame(X, columns=X_columns)

# diviser les features et label en ensembles d’entraînement et en ensembles de test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=34)
x_train.shape, x_test.shape

y_train.shape, y_test.shape

# 
svc = SVC()
svc.fit(x_train, y_train)
svc_prediction = svc.predict(x_test)

svc_score = accuracy_score(y_test, svc_prediction)
svc_score

