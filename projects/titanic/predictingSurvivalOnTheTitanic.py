#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import accuracy_score
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[2]:


train        = pd.read_csv('Titanic_train.csv')
test         = pd.read_csv('Titanic_test.csv')
passengerid  = test['PassengerId']


# In[3]:


train.head()


# In[4]:


train.shape


# In[5]:


train.isnull().sum()


# In[6]:


test.head()


# In[7]:


test.shape


# In[8]:


test.isnull().sum()


# 'Name' est inutile est 'Cabin' manque de beaucoup de valeurs, elles seront supprimées.

# In[9]:


train = train.drop(columns=['Name', 'Cabin'])
test  = test.drop(columns= ['Name', 'Cabin'])


# In[10]:


# Remplacer les âges manquants par la moyenne `fullna()`
train_age_mean = train['Age'].mean()
test_age_mean  = test['Age'].mean()

train.fillna({'Age':train_age_mean}, inplace=True)
test.fillna ({'Age':train_age_mean}, inplace=True)


# In[11]:


# Remplacer les 2 valeurs manquantes d'"embarled" par la valeur la plusprésenteabs
train.fillna({'Embarked':train['Embarked'].mode()[0]}, inplace=True)


# In[12]:


# Supprimer 'PassengerId' ; inutile
train.drop(columns='PassengerId', inplace=True)
test.drop (columns='PassengerId', inplace=True)


# ### Exploration des données

# In[13]:


Pclass_count = train.Pclass.value_counts()
Pclass_count.plot(kind='bar', xlabel='Pclass', ylabel='Count')


# La majorité était en troisième classe.

# In[14]:


sex_count = train.Sex.value_counts()
sex_count.plot(kind='bar', xlabel='Sex', ylabel='Count')


# Et deux fois plus d'hommes.

# In[15]:


# Combien de personne ont survécu ?
Survived_count = train.Survived.value_counts()
Survived_count.plot(kind='bar', xlabel='Survived', ylabel='Count')


# In[16]:


# Combien sont morts ?
sns.countplot(train, x='Sex', hue='Survived')


# In[17]:


# Quelle classe a eu le plus de survivants ?
sns.countplot(train, x='Pclass', hue='Survived')


# La plupart des morts étaient en troisième classe. La richesse est un avantage.

# In[18]:


# Convertir les valeurs des colonnes catégorielles en nombres
train['Sex'] = train['Sex'].replace({'male':1, 'female': 0})
test['Sex']  = test ['Sex'].replace({'male':1, 'female': 0})

train['Embarked'] = train['Embarked'].replace({'S':2, 'C': 1, 'Q':0})
test['Embarked']  = test ['Embarked'].replace({'S':2, 'C': 1, 'Q':0})


# In[19]:


# Supprimer la colonne 'Ticket'
train.drop(columns='Ticket', inplace=True)
test.drop (columns='Ticket', inplace=True)


# In[20]:


# Diviser en featured et labels
y = train.Survived.copy()
x = train.drop(columns='Survived').copy()


# In[21]:


# Pour avoir des données pour tester les performances des modèles, diviser ces données en ensembles d’entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=786)


# #### Classificateur Decision Tree

# In[22]:


dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

dtc_prediction = dtc.predict(x_test)

dtc_score = accuracy_score(y_test, dtc_prediction)
dtc_score


# Correct.

# #### Classificateur Random Foresst

# In[23]:


rfc = RandomForestClassifier(max_depth=15)
rfc.fit(x_train, y_train)

rfc_prediction = rfc.predict(x_test)

rfc_score = accuracy_score(y_test, rfc_prediction)
rfc_score


# Mieux

# #### Régression logique

# In[24]:


lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_prediction = lr.predict(x_test)
lr_score      = accuracy_score(y_test, lr_prediction)
lr_score


# Moyen

# #### Améliorer le classificateur Random Forest

# In[25]:


param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfc = RandomForestClassifier()

grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print("Meilleurs hyperparamètres :", best_params)

best_rfc = RandomForestClassifier(**best_params)
best_rfc.fit(x_train, y_train)
rfc_prediction = best_rfc.predict(x_test)
rfc_score = accuracy_score(y_test, rfc_prediction)
print("Score d'accuracy :", rfc_score)


# In[26]:


test.fillna({'Fare':test['Fare'].mean()}, inplace=True)


# In[27]:


# Utiliser ce modèle et obtenir les prédictions pour les données de test
my_prediction = best_rfc.predict(test)

submission_df = pd.DataFrame({'PassengerId':passengerid, 'Survived':my_prediction})
submission_df.to_csv('submission.csv', index=False)

