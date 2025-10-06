# -*- coding: utf-8 -*-

# -- Sheet --

import pandas                  as     pd
from   sklearn.model_selection import train_test_split
from   sklearn.tree            import DecisionTreeClassifier
from   sklearn.metrics         import accuracy_score
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.model_selection import GridSearchCV
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.ensemble        import RandomForestClassifier

df = pd.read_csv('diabetes.csv')

df.shape


df.isnull().sum()

df.info()

df.hist(figsize=(15, 10))

# Diviser le df en train de test
y = df['Outcome']
X = df.drop(columns='Outcome')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=84)

# Classifier par arbre de décision
# Mise à l'échelle est normaliser les features pour l'arbre décisionnel
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_prediction = dtc.predict(X_test)

dtc_accuracy = accuracy_score(y_test, dtc_prediction)
dtc_accuracy

# Combiner plusieurs arbres en forêt décisionnelle
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)
rfc_score = accuracy_score(y_test, rfc_prediction)
rfc_score

# Améliorer l'arbre décisionnel
# Définir la grille d'hyperparamètres
param_grid = {
    'n_estimators'     : [100, 200, 300],   # Nombre d'arbres dans la forêt
    'max_depth'        : [None, 10, 20],    # Profondeur maximale des arbres
    'min_samples_split': [2, 5, 10],        # Nombre minimum d'échantillons requis pour séparer un nœud interne
    'min_samples_leaf' : [1, 2, 4]          # Nombre minimum d'échantillons requis pour être à un nœud feuille
}
# Instancier le classificateur de la RandomForest
rf = RandomForestClassifier()
# Instanciation de la grid search
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, # essai tous les param. par param_grid. et donnera la meilleure
                           cv = 3, n_jobs = -1, verbose = 2)
# Ajuster la grid search aux données
grid_search.fit(X_train, y_train)
# Obtenir les meilleurs paramètres
best_params = grid_search.best_params_

# Quels sont les meilleurs paramètres ?
best_params

rfc_updated = RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
rfc_updated.fit(X_train, y_train)
rfc_updated_prediction = rfc_updated.predict(X_test)
rfc_updated_score = accuracy_score(y_test, rfc_updated_prediction)
rfc_updated_score

