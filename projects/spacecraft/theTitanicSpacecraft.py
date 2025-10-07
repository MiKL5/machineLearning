# -*- coding: utf-8 -*-

# -- Sheet --

import pandas                  as     pd
from   sklearn.model_selection import train_test_split
from   sklearn.tree            import DecisionTreeClassifier
from   sklearn.metrics         import accuracy_score, confusion_matrix
from   sklearn.ensemble        import RandomForestClassifier
from   xgboost                 import XGBClassifier
from   sklearn.ensemble        import StackingClassifier
from   sklearn.linear_model    import LogisticRegression
from   sklearn.svm             import SVC

# # **Vaisseau spatial Titanic**


train       = pd.read_csv('SpaceshipTitanic_Train.csv')
test        = pd.read_csv('SpaceshipTitanic_Test.csv')
passengerid = test["PassengerId"]

train.shape, test.shape

train.head()

train.isnull().sum()

# Il manque des valeurs dans beaucoup de colonnes.


test.isnull().sum()

# Idem.


# ## **Traiter les valeurs manquantes**


train.head(30)

train.tail(30)

# La précédante valeur est répétée à la plupart des lignes.


train.drop(columns=['Name', 'PassengerId'], inplace=True)
test .drop(columns=['Name', 'PassengerId'], inplace=True)

# La méthode `fill` comlète par la valeur précédante
train[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']] = train[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']].ffill()
test[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']] = test[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']].ffill()

train.isnull().sum()

test.isnull().sum()

# Beaucoup de colonnes ont des valeurs manquantes.


# `Cabin` et `CabinNum` sont inutiles
train.drop(columns=['Cabin'] , inplace=True)
test.drop(columns =['Cabin'] , inplace=True)

train.hist(figsize=(20, 10))

# Compléter par l'âge moyen. Idem pour RoomService, FoodCart, ShoppingMall, Spa, VRDeck
train['Age']          = train['Age'].fillna(train['Age'].mean())
test['Age']           = test['Age'].fillna(test['Age'].mean())
train['RoomService']  = train['RoomService'].fillna(train['RoomService'].mode()[0])
test['RoomService']   = test['RoomService'].fillna(test['RoomService'].mode()[0])
train['FoodCourt']    = train['FoodCourt'].fillna(train['FoodCourt'].mode()[0])
test['FoodCourt']     = test['FoodCourt'].fillna(test['FoodCourt'].mode()[0])
train['ShoppingMall'] = train['ShoppingMall'].fillna(train['ShoppingMall'].mode()[0])
test['ShoppingMall']  = test['ShoppingMall'].fillna(test['ShoppingMall'].mode()[0])
train['Spa']          = train['Spa'].fillna(train['Spa'].mode()[0])
test['Spa']           = test['Spa'].fillna(test['Spa'].mode()[0])
train['VRDeck']       = train['VRDeck'].fillna(train['VRDeck'].mode()[0])
test['VRDeck']        = test['VRDeck'].fillna(test['VRDeck'].mode()[0])

train.isnull().sum()

test.isnull().sum()

# Il n'y a plus de valeur manquante.  
# Convertir les colonnes catégorielles et non numériques.


train.dtypes

# ## **Convertir les colonnes catégorielles et les colonnens non numériques**


# Convertir les boulléens en entiers
train[['CryoSleep', 'VIP', 'Transported']] = train[['CryoSleep', 'VIP', 'Transported']].astype(int)
test[['CryoSleep', 'VIP']] = test[['CryoSleep', 'VIP']].astype(int)

# Utiliser la technique “One-Hot Encoding“
df = pd.get_dummies(train, columns=['HomePlanet', 'Destination'])
pd.set_option('display.max_columns', 100)
df

# `One-Hot Encoding` est une technique formate les variables en binaire afin que les modèles de machine learning puissent les utiliser efficacement.
# 
# Le principe de base
# Pour chaque catégorie possible d'une variable, on crée une nouvelle colonne (ou « feature »). Dans chaque ligne, la colonne correspondant à la catégorie présente prend la valeur 1, et toutes les autres prennent 0. Afin de ne pas introduire d'ordre artificiel entre les catégories.


# ## **Construire les mocèles et prédicitons**


# Répartir
y = df['Transported'].copy()
X = df.drop(columns='Transported').copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# ### **Classifier par un arbre décisionnel**


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_prediction = dtc.predict(X_test)
dtc_accuracy = accuracy_score(y_test, dtc_prediction)
dtc_accuracy

dtc_confusion_matrix = confusion_matrix(y_test, dtc_prediction)
dtc_confusion_matrix

# ## **Classifier avec random forest**


rfc = RandomForestClassifier(random_state=786)
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)
rfc_accuracy = accuracy_score(y_test, rfc_prediction)
rfc_accuracy

# ### **Classificateur XGB**


xgbc = XGBClassifier(random_state=786)
xgbc.fit(X_train, y_train)
xgbc_prediction = xgbc.predict(X_test)
xgbc_accuracy = accuracy_score(y_test, xgbc_prediction)
xgbc_accuracy

# Le classificateur en forêt a le même résultat.


# ### **Approche pour le stacking**


# Modèles de base
estimators = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
]

# Méta-modèle
final_estimator = LogisticRegression()

# Modèle stacking
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

# Entraînement
stacking_clf.fit(X_train, y_train)

# Prédiction et évaluation
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision stacking 👉 {accuracy}")

# L'appreche par empilement a fait à peine mieux.  
# C'est une méthode d'ensemble learning qui combine plusieurs modèles de machine learning appelés modèles de base pour créer un modèle final plus performant.  
# 
# ⟹ Le stacking permet d'apprendre comment combiner plusieurs modèles imparfaits mais complémentaires pour créer un modèle final plus puissant et général.


