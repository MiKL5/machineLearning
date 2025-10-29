# **Prédire le score de dépences des clients**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&logo=python&logoColor=FFD) 
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Framework-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat&logo=matplotlib&logoColor=white) 
![Seaborn](https://img.shields.io/badge/seaborn-Statistical_Visualization-556F9F?style=flat&logo=python&logoColor=white)

</div><hr>

Ce projet présente une analyse prédictive du score de dépenses (Spending Score) des clients d’un centre commercial à partir d’un dataset comprenant notamment le revenu annuel et le genre.
Différents modèles de régression sont comparés afin d’identifier la meilleure approche pour estimer ce score.
---
## **Le prétraitement**
* Renommer les colonnes pour simplifier leur utilisation.  
* Séparer les variables explicatives (x) et cible (y = Spending Score).  
* Encoder la variable catégorielle Gender en valeurs numériques (0/1).  
* Découper les données en ensembles d'entraînement (80%) et de test (20%) pour évaluer les performances.
## **La modélisation et l'évaluation**
* Support Vector Regression (SVR)
* Arbre de Décision (Decision Tree Regressor)
* Régression Linéaire
* Test de normalisation via StandardScaler sur les variables explicatives et évaluation de l’impact sur SVR.
## **Les résultats**
Les arbres de décision obtiennent une meilleure erreur absolue moyenne (MAE) que les autres modèles testés.  
La normalisation par standardisation n'améliore pas significativement la prédiction sur ce dataset spécifique.  
Les performances globales restent modestes, suggérant que d’autres variables ou modèles plus complexes seraient nécessaires.
## **Nota**
L’analyse comparative illustre l’importance du choix de modèle en prédiction.  
D’autres variables explicatives ou méthodes plus avancées peuvent améliorer la précision.  
Ce projet sert d’introduction aux bases de la modélisation prédictive de scores d’engagement client.