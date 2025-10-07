# **Le vaisseau spacial Titanic**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
![XGBoost](https://img.shields.io/badge/xgboost-Gradient_Boosting-FF6600?style=flat&logo=xgboost&logoColor=white) 
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div>

___
## **L'objectif**
Prédire la survie des passagers du vaisseau spatial Titanic à partir de données numériques et catégorielles issues de divers paramètres personnels et environnementaux.  
Il utilise des techniques avancées de machine learning supervisé pour modéliser les probabilités de survie.
---
## **Le jeu de données**
Les données se composent des informations issues des passagers du Titanic Spacecraft :  
* Caractéristiques démographiques : planète d'origine, âge, etc.  
* Statuts spécifiques : cryosommeil, VIP, destination.  
* Données de dépenses à bord : services divers (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck).  
* Cible à prédire : transporté (survie ou non).

Des valeurs manquantes sont largement présentes et traitées dans le pipeline.
## **La méthodologie**
### **La préparation des données**
* Suppression des colonnes non-informative (`Name`, `PassengerId`, `Cabin`).
* Imputation des valeurs manquantes par méthode appropriée :
  * Remplissage par valeurs précédentes pour colonnes catégorielles continues (ex : `HomePlanet`, `CryoSleep`).
  * Moyenne ou mode pour les colonnes quantitatives.
* Transformation des variables catégorielles par One-Hot Encoding.
* Conversion des booléens en entiers.
### **La modélisation**
* Séparation du jeu en ensembles entraînement (80%) et test (20%).  
* Entraînement, évaluation et comparaison de plusieurs modèles :  
  * Arbre de décision (`DecisionTreeClassifier`)  
  * Forêt aléatoire (`RandomForestClassifier`)  
  * XGBoost (`XGBClassifier`)  
* Mise en place d'un méta-modèle par empilement (Stacking) avec comme estimateur final la régression logistique (`LogisticRegression`).
### **L'évaluation**
* Précision (`accuracy`) sur le jeu test calculée pour chaque modèle.
* Matrice de confusion pour l’arbre de décision afin d’analyser les classifications.
## **Les résultats clés**
* Le modèle par empilement (stacking) améliore la précision comparé aux modèles simples.  
* XGBoost se révèle performant pour la classification binaire dans ce contexte.  
* Le nettoyage et l’imputation jouent un rôle crucial dans la qualité des prédictions.