# Prédiction des prix immobiliers à Melbourne avec apprentissage automatique <a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=FFD43B) 
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-FF9900?style=flat&logo=scikit-learn&logoColor=white) 
![Pandas](https://img.shields.io/badge/pandas-Data_Processing-150458?style=flat&logo=pandas&logoColor=white) 

</div><hr>

Ce projet illustre l’utilisation de modèles de régression supervisée pour prédire le prix des propriétés immobilières à Melbourne à partir du dataset `melbourne_data.csv`.  
Le pipeline intègre le nettoyage des données, la transformation des variables catégorielles, la sélection de features pertinentes, et l’évaluation des performances des modèles.
---
## Le dataset
* **Taille** : 13 580 observations, 21 colonnes  
* **Colonnes clés** :  
  * Catégorielles : `Suburb`, `Address`, `Type`, `Method`, `SellerG`, `Date`, `Regionname`  
  * Numériques : `Rooms`, `Price` (cible), `Distance`, `Postcode`, `Bedroom2`, `Bathroom`, `Landsize`, `Lattitude`, `Longtitude`, `Propertycount`  
* Les colonnes ayant trop de valeurs manquantes sont supprimées : `BuildingArea`, `YearBuilt`, `CouncilArea`  
* Pour éviter d'introduire du bruit, supprimer `Car`.
## Prétraiter les données
1. Suppreimer les colonnes à trop forte proportion de valeurs manquantes.  
2. Identifier les colonnes catégorielles à faible cardinalité (`Type`, `Method`, `Regionname`).  
3. Coder les variables catégorielles avec `LabelEncoder` pour transformer les catégories en entiers sans ordre implicite.  
4. Séparer les features (X) et de la cible (y = `Price`).  
5. Découper les données en ensembles d’entraînement et de test (80% / 20%).
## Modéliser
* Tests de deux modèles :  
  * **DecisionTreeRegressor** : MAE élevé (~224 820), performance insuffisante.  
  * **RandomForestRegressor** : meilleure MAE (~170 211), amélioration notable grâce à l’ensemble d’arbres.
## **Le résultat**
Le modèle Random Forest fournit une meilleure estimation du prix, malgré une marge d'erreur encore conséquente.
## **La conclusion**
* Le nettoyage rigoureux des données et la sélection judicieuse des variables sont cruciaux pour des prédictions robustes.
* Les modèles d'arbres ensemblistes (Random Forest) surpassent les modèles d’arbres simples dans ce contexte.
* Il est recommandé d’explorer d’autres algorithmes, techniques de tuning et d’ingénierie des features pour améliorer la précision.