# Prédiction de la qualité du vin avec apprentissage automatique <a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=FFD43B) 
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Classification-FF9900?style=flat&logo=scikit-learn&logoColor=white) 
![XGBoost](https://img.shields.io/badge/xgboost-Gradient_Boosting-FF6600?style=flat&logo=xgboost&logoColor=white) 
![Seaborn](https://img.shields.io/badge/Seaborn-Data_Visualization-556F9F?style=flat&logo=python&logoColor=white) 

</div><hr>

Ce projet illustre un flux complet de **préparation des données**, d'**analyse statistique**, et de **modélisation prédictive** de la qualité du vin.  
Les scripts sont conçus pour explorer la distribution des variables, gérer les valeurs manquantes, encoder les variables catégorielles et comparer plusieurs modèles de classification.
---
## **Le jeu de données**
* **Taille** : 6497 entrées, 13 colonnes  
* **Type** : mélange de variables numériques et d’une catégorie (`type`)  
* **Objectif** : prédire la qualité du vin (`best quality`, 0 ou 1) selon des caractéristiques physico-chimiques.
### **Les variables principales**
* Acides : `fixed acidity`, `volatile acidity`, `citric acid`
* Teneurs : `residual sugar`, `chlorides`, `sulphates`
* Facteurs chimiques : `density`, `pH`, `alcohol`
* Mesures du dioxyde de soufre : `free sulfur dioxide`, `total sulfur dioxide`
* Cible : `best quality` (1 si qualité > 5, sinon 0)
## **Préparer les données**
1. **Le nettoyage**  
   * Remplacer les valeurs manquantes par la moyenne de chaque colonne.  
   * Supprimer la colonne originale `quality`.
2. **L'encodage**  
   * Transformer la variable `type` en étiquettes numériques via `LabelEncoder`.
3. **La normalisation**  
   * Appliquer de `MinMaxScaler` pour uniformiser les distributions numériques.
4. **Le découpage**  
   * Entraîner et tester : 80 % / 20 % avec `train_test_split`.
## L'exploration visuelle
* **L'histogrammes** : distribution globale des variables.
* **La corrélation** : visualisation du lien entre variables avec une heatmap Seaborn.  
La variable `alcohol` montre la plus forte corrélation avec la qualité.
## **L'explicabilité des modèles**
* Visualiser les **importances des variables** pour identifier les features les plus influents.
* Utiliser **SHAP values** (SHapley Additive exPlanations) pour expliquer la contribution de chaque variable à chaque prédiction.
* Ces méthodes aident à valider que le modèle se base sur des facteurs cohérents avec la connaissance métier.
## **Les résultats**
* Le modèle XGBoost s'impose par sa meilleure précision pour cette classification binaire.
* L'alcool est la variable la plus déterminante pour la qualité prédite.  
* Le préprocessing rigoureux améliore la robustesse des modèles.  
* Le pipeline prêt à l’usage est adaptable à d’autres jeux de données tabulaires.