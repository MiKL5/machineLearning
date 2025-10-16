<h1><a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a><b>Prédiction du prix des maisons en Californie – Modèles<hr>
Random Forest et Arbre de Décision</b></h1>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b)
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)

</div>

___
## **Le projet**
Ce projet implémente un pipeline complet pour prédire le prix de vente median des maisons en Californie à l'aide de modèles de machine learning supervisé : **DecisionTreeRegressor** et **RandomForestRegressor**.  
Après un pré-traitement robuste des données, les modèles sont entraînés sur un dataset d'entraînement et évalués sur un ensemble test séparé.
---
## **Le jeu de données**
- `HousingPriceTrain.csv` : données d'entraînement avec la cible `SalePrice`.
- `HousingPriceTest.csv` : données de test sans variable cible pour validation finale.
Le dataset contient un grand nombre de colonnes, avec plusieurs valeurs manquantes et variables catégorielles.
## **Les étapes et la méthodologie**
### **Nettoyage et préparation**
* Identification et suppression des colonnes à forte proportion de valeurs manquantes (ex : `Alley`, `PoolQC`, `Fence`...).
* Suppression des lignes restantes contenant des valeurs manquantes.
* Séparation des colonnes en :  
  * Catégorielles à faible cardinalité (< 10 valeurs uniques)  
  * Catégorielles à haute cardinalité (≥ 10 valeurs uniques)  
  * Numériques  
* Encodage des colonnes catégorielles avec `LabelEncoder`.
### **La division du jeu d'entraînement**
* Séparation en données d'entrée (`X`) et cible (`SalePrice`).
* Split entraînement/test (80/20) via `train_test_split` avec graine pour reproductibilité.
### **Les modèles testés**
* **DecisionTreeRegressor** : modèle de base non paramétré, robuste aux données brutes.
* **RandomForestRegressor** : ensemble d’arbres aléatoires pour meilleure généralisation.
### **L'évaluation**
* Les métrics utilisées :  
  * `mean_absolute_error` (MAE)  
  * `mean_absolute_percentage_error` (MAPE)
* Comparaison des performances sur l’ensemble test.
---
## **Les résultats obtenus**
* Les deux modèles fonctionnent bien, avec une meilleure précision obtenue par la forêt aléatoire (`RandomForestRegressor`).
* Le pré-traitement des données (nettoyage, encodage) a été déterminant pour ces performances.
<!-- ## Licence
Sous licence MIT — voir fichier [LICENSE](LICENSE) pour les détails. -->
___
## **Notes**
* Projet illustrant un pipeline classique de machine learning supervisé appliqué à de la régression.  
* Pré-traitement des données et encodage simple mais efficace.  
* Modèles classiques avec bonne performance sans réglages hyperparamétriques détaillés.  
* Approche aisée à adapter à d'autres jeux de données similaires.