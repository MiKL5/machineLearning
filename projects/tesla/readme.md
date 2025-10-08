# Prédire le prix de l'action Tesla<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat&logo=matplotlib&logoColor=white) 
![Seaborn](https://img.shields.io/badge/seaborn-Statistical_Visualization-556F9F?style=flat&logo=python&logoColor=white) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Framework-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div><hr>

## **La description**
Analyser, visualiser et prédire l’évolution du prix de l’action Tesla de 2014 à 2023.  
Une démarche professionnelle combinant data engineering, visualisation avancée et modélisation machine learning a été appliquée. Allant du nettoyage jusqu’à la comparaison robuste de plusieurs modèles de régression.
---
## **Pipeline d'analyse & modélisation**
### **La préparation des données**
* Lecture et transformation des données historiques (`tesla20142023.csv`)
* Conversion de la colonne `date` au format datetime et utilisation comme index temporel
* Détection et visualisation des valeurs manquantes
* Analyse exploratoire (statistiques & heatmaps de corrélation)
### La visualisation
* Représentation interactive des prix : `Open`, `Close`, `High`, `Low`
* Matrice de corrélation pour détecter les relations clés entre variables
### **Le Feature Engineering**
* Séparation des features (`X`) et de la cible (`nextdayclose`)
* Split `train/test` (80/20) pour validation robuste
* Scaling des features via `StandardScaler` pour garantir des modèles bien calibrés
### **La modélisation**
* Les modèles testés
  * Régression Linéaire (`LinearRegression`)
  * Arbre de Décision (`DecisionTreeRegressor`)
  * Forêt Aléatoire (`RandomForestRegressor`)
  * Support Vector Regressor (`SVR`)
* Évaluation comparée sur la base du **Mean Absolute Error (MAE)**
___
## **Résultats attendus et Enhancements**
* Comparaison systématique des performances pour un choix objectif du meilleur modèle de prédiction (MAE)
* Visualisation claire du comportement et de la tendance des prix
* **Améliorations possibles :**
    * Intégration d’indicateurs techniques (RSI, MACD, etc.)
    * Gestion automatisée de la mise à jour des données depuis une API
<!-- ___
## Licence
Code sous licence MIT : réutilisation, modification et diffusion libres avec attribution. -->
___
### **Remarques**
- Pour toute utilisation en production ou backtesting réel, valider la qualité des fichiers de données sources et la stabilité du pipeline.
- Crédits : projet à visée pédagogique (Master Big Data/IA).