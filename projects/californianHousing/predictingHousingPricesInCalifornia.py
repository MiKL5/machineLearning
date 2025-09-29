#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.linear_model    import LinearRegression
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error, mean_absolute_error
from sklearn.tree            import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble        import RandomForestRegressor
import joblib


# # Prédiction **des prix du logement en Californie*

# In[2]:


df = pd.read_csv('california_housing.csv')


# In[3]:


df.head()


# In[4]:


df.tail()                  # Les 5 dernières


# In[5]:


df.shape                    # La forme du dataframe


# In[6]:


df.isnull().sum()           # Les valeurs manquantes


# In[7]:


df.info()                    # Les types de données


# In[8]:


df.describe()                # Affiche `count`, `mean`, `std`, `min`, `max`, `perrcentiles`


# In[9]:


df['ocean_proximity'].unique()  # différentes valeurs


# In[10]:


df['ocean_proximity'].value_counts()         # nombres d'occurences par valeur


# In[11]:


df.hist(bins=30, figsize=(20, 10))          # dispersion des valeurs


# ### Diviser le dataframe

# In[12]:


train_set, test_set = train_test_split(df, test_size=0.2, random_state=8)
train_set.shape, test_set.shape


# In[13]:


df['median_income'].min(), df['median_income'].max()


# In[14]:


df['income_cat'] = pd.cut(df['median_income'], bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])


# In[15]:


# diviser le DataFrame df en un jeu d'entraînement et un jeu de test (80% / 20%) tout en conservant la même répartition proportionnelle des catégories présentes dans la colonne income_cat dans les deux sous-ensembles
strat_train_set, strat_test_set = train_test_split(df, test_size=0.2, stratify=df['income_cat'], random_state=8)


# In[16]:


strat_train_set['income_cat'].value_counts()


# In[17]:


strat_test_set['income_cat'].value_counts()


# Ce que cela signifie
# * La stratification sur `income_cat` a permis de conserver des proportions similaires de chaque catégorie dans le jeu d'entraînement `(strat_train_set)` et dans le jeu de test `(strat_test_set)`.
# * E. g. la catégorie 3 représente 5789 occurrences dans l'entraînement et 1447 dans le test, soit environ 20 % dans le test par rapport à l'entraînement, conforme au paramètre `test_size=0.2`.
# * La répartition homogène évite les biais statistiquement importants qui pourraient survenir avec un découpage purement aléatoire. Cela garantit une bonne représentativité des différentes populations dans les deux sous-jeux.
# * Cette homogénéité est primordiale pour entraîner un modèle fiable et pour évaluer ses performances de manière réaliste, surtout en présence de classes moins fréquentes (ici la classe 1 est la moins représentée, mais bien présente dans train et test).
# 
# L'importance :
# * Si la distribution avait été différente entre train/test, les performances sur le test ne seraient pas représentatives lors du déploiement.
# * La stratification préserve ainsi la validité statistique des estimations issues du test.
# 
# La fonction `train_test_split` avec `stratify` a correctement réparti les classes `income_cat` en proportions proches dans les jeux train et test, garantissant l’équilibre nécessaire pour un apprentissage et une validation robustes.

# In[18]:


# suppression de la colonne `income_cat`
strat_train_set.drop(columns='income_cat', inplace=True)
strat_test_set.drop(columns='income_cat', inplace=True)


# In[19]:


df.plot(kind='scatter', x='longitude', y='latitude', figsize=(10, 6), 
             s=df['population']/100, label='population', alpha=0.1, 
             cmap='jet', c='median_house_value', colorbar='True')


# ## **Préparer les donnes pour les algos de ML**

# ### Diviser l'ensemble de données en features et label

# In[20]:


housing = strat_train_set.drop(columns='median_house_value')
housing_label = strat_train_set['median_house_value'].copy()


# In[21]:


housing_num = housing.drop(columns='ocean_proximity')   # Suppresssion ; il s'agit d'un pipelin de données numériques


# `SimpleImputer(strategy='median')` : remplace les valeurs manquantes de chaque colonne par la médiane correspondante. Cette stratégie est robuste aux valeurs extrêmes.
# 
# `StandardScaler()` : standardise les variables numériques en soustrayant la moyenne et en divisant par l'écart-type, ce qui donne des données centrées réduites avec moyenne 0 et écart-type 1. Cela facilite la convergence de nombreux algorithmes d'apprentissage.
# 
# Pipeline : chaîne les transformations pour pouvoir appliquer facilement la suite au DataFrame, offrant une organisation propre et reproductible.
# 
# `fit_transform(housing_num)` : calcule les statistiques nécessaires (médianes, moyennes, écarts-types) durant l'ajustement et transforme les données en une matrice NumPy normalisée.

# In[22]:


num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('standarize', StandardScaler())
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr


# In[23]:


# définir les listes des noms des colonnes numériques et catégoriques à traiter séparément dans un pipelineabs
num_attributes = list(housing_num)
cat_attributes = ['ocean_proximity']


# In[24]:


# transformer deux groupes de colonnes et transformation du `full_pipeline` au df complet, combinant proprement traitement numérique et encodage catégoriel
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', OneHotEncoder(), cat_attributes)
])


# In[25]:


housing_prepared = full_pipeline.fit_transform(housing) # nettoyer et pré-traiter le dataset d’origine via le pipeline


# In[26]:


lr = LinearRegression()                                       # Créer un  modèle de régression linéaire
lr.fit(housing_prepared, housing_label)                       # Ajuster les coefficients de la régression linéaire pour minimiser l'erreur quadratique entre les prédictions et la variable cible à partir des données prétraitées
some_data = housing.iloc[:5]                                  # Récupèrer les 5 premiers exemples à prédire
some_label = housing_label.iloc[:5]                           # Conserver les étiquettes pour évaluer
some_prepared_data = full_pipeline.transform(some_data)       # Transformer ces données brutes avec le pipeline sans recalculer les paramètres déjà appris
some_prediction = lr.predict(some_prepared_data)              # Prédire les valeurs cibles selon la relation linéaire apprise, pour les 5 exemples préparés


# In[27]:


# Calculer l’erreur quadratique moyenne (MSE) entre les valeurs réelles et les valeurs prédites
# Prendre la racine carrée de cette erreur pour obtenir la racine de l’erreur quadratique moyenne (RMSE).
error = mean_squared_error(some_label, some_prediction)
np.sqrt(error)


# In[28]:


dtr = DecisionTreeRegressor()                               # Créer un modèle de régression par arbre décisionnel
dtr.fit(housing_prepared, housing_label)                    # Entrainer le modèle sur les données prétraitées et leurs étiquettes (valeurs cibles)
dtr_prediction = dtr.predict(some_prepared_data)            # Prédir la valeur cible pour de nouvelles données préparées
dtr_error = mean_squared_error(some_label,dtr_prediction)   # Calculer l’erreur quadratique moyenne entre les vraies valeurs et les prédictions du modèle
np.sqrt(dtr_error)                                          # RMSE, mesurant l’erreur moyenne en unités d’origine.


# In[29]:


scores = cross_val_score(dtr, housing_prepared, housing_label, scoring='neg_mean_squared_error', cv=10)
np.sqrt(-scores)                   # RMSE pour chaque fold


# In[30]:


rfr = RandomForestRegressor()              # Créer un modèle de forêt aléatoire pour une tâche de régression, composé de multiples arbres de décision moyennés pour améliorer stabilité et précision
rfr.fit(housing_prepared, housing_label)   # Entraîner ce modèle sur les données prétraitées et les étiquettes associées
rfr_scores = cross_val_score(rfr, housing_prepared, housing_label, scoring='neg_mean_squared_error', cv=10) # Faire une validation croisée à 10 folds
np.sqrt(-rfr_scores), np.sqrt(-rfr_scores).mean()       # Convertir les scores de validation croisée en RMSE, calculer leur moyenne pour évaluer globalement la performance du modèle Random Forest


# In[31]:


joblib.dump(rfr, 'rfr_model.pkl')   # Persister le modèle dans un fichier

