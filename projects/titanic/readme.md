# **Prédire la survie sur le Titanic**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">
  <img src="https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b" alt="Python" />
  <img src="https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/seaborn-Data_Visualization-0074D9?style=flat&logo=seaborn&logoColor=white" alt="Seaborn" />
</div>

## **L'objectif du projet**
Ce projet a pour but de développer un modèle de machine learning capable de prédire si un passager donné du Titanic a survécu ou non, basé sur ses caractéristiques personnelles et son contexte de voyage.
---
## **Le jeu de données**
Le jeu de données émane de Kaggle et contient :
* L'identifiant du passager (PassengerId) ;
* La classe de billet (Pclass) ;
* Les Nom et Sexe (Sex) et âge (Age) ;
* Le nombre de frères/sœurs à bord (SibSp) ;
* le nombre de parents/enfants à bord (Parch) ;
* Le numéro de billet (Ticket) ;
* Le tarif payé (Fare) ;
* La cabine (Cabin) ;
* Le port d'embarquement (Embarked) ;
* L'étiquette cible : survie (Survived),

Certaines colonnes ayant trop de valeurs manquantes ou peu pertinentes (e.g. Nom, Cabine, Ticket) ont été supprimées.
Les valeurs manquantes d'âge et d'embarquement ont été imputées avec la moyenne ou la valeur la plus fréquente.
___
## Les principales étapes
### **1. Pré-traitement**
* Le nettoyage et suppression des colonnes inutiles
* L'imputation des valeurs manquantes
* La transformation des variables catégorielles en variables numériques (encodage)
* La séparation des données en features et étiquette cible
### **2. L'exploration**
* L'analyse des répartitions de classes, sexe, survie
* La visualisations graphiques des tendances et corrélations (avec Seaborn, matplotlib)
### **3. La construction des modèles**
* Décision Tree Classifier
* Random Forest Classifier
* Logistic Regression
### **4. L'optimisation**
* Recherche des meilleurs hyperparamètres pour le Random Forest via GridSearchCV
### **5. L'évaluation**
* Évaluation des performances avec accuracy score sur un jeu de test (validation)
### **6. La prédiction finale**
* Application du meilleur modèle sur les données de test non étiquetées
* Sauvegarde des résultats de prédiction dans `submission.csv`
___
## **Les résultats**
* Le modèle Random Forest optimisé a obtenu la meilleure accuracy ;
* La prise en compte des hyperparamètres a permis d'améliorer notablement la précision ;
* La visualisation a confirmé que le sexe et la classe étaient des facteurs fortement corrélés à la survie.