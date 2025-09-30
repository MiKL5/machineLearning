#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas                as pd
import matplotlib.pyplot     as plt
from sklearn.tree            import DecisionTreeClassifier 
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import accuracy_score


# In[2]:


df = pd.read_csv('Heart_Disease_Prediction.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


# Remplacer absence et présence par 0 et 1
df['Heart Disease'] = df['Heart Disease'].replace({'Presence': 1, 'Absence': 0}).astype(int)


# In[7]:


# Pour mettre à l'échelle et normaliser les features, il faut fractionner
y = df['Heart Disease']
X = df.drop(columns='Heart Disease')


# In[8]:


# Visualisation après normalisation
plt.figure(figsize=(8,6))
scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('Scatter plot des deux premières features coloré par Heart Disease')
plt.colorbar(scatter, label='Heart Disease (0=Absence, 1=Présence)')
plt.show()


# In[9]:


# Validation croisée
dtc_scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5, scoring='accuracy')
print(dtc_scores)
print(dtc_scores.mean())


# In[10]:


# Essai avec Random Forest
models = [DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier()]
for model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(model, scores.mean())


# Bonne accuracy.

# In[11]:


# Améliorer l' exactitude en ajoutant des hyperparamètres avec l'outil `GridSearchCV`
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
grid_search.best_estimator_


# In[12]:


# Il recommande `max_depth=10` pour amèliorer l'accuracy
rfc = RandomForestClassifier(max_depth=10)
rfc.fit(X_train, y_train)

rfc_prediction = rfc.predict(X_test)

rfc_score = accuracy_score(y_test, rfc_prediction)
rfc_score


# La précision est inférieure à 89 %.  
# Ce n'est pas mauvais.
