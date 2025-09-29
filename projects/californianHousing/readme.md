# **Prédire les prix des Logements en Californie**<a href="../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b)
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-Scientific_Computing-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat&logo=matplotlib&logoColor=white)

</div>

Ce projet Python utilise des techniques de machine learning pour prédire les prix des logements en Californie à partir d’un jeu de données complet et varié. L'accent est mis sur la préparation des données, la modélisation et l’évaluation rigoureuse des performances.
## **L'objectif**
Prédire la valeur médiane des logements en fonction de critères géographiques, démographiques et économiques, afin de mieux comprendre les facteurs influençant les prix et d’évaluer la qualité des modèles prédictifs.
## **Le jeu de données**
Le dataset utilisé provient d'une source ouverte*, contenant des informations sur :
* La localisation (longitude, latitude)
* Le revenu médian des ménages
* La taille des logements
* Le nombre de pièces et de chambres
* La proximité à l'océan
* La population locale
* La valeur médiane des logements (variable cible)
___
## **La méthodologie**
### **L'exploration des données**
* Inspection initiale (dimensions, valeurs manquantes, types)
* Visualisation des distributions et des corrélations
* Analyse des catégories qualitatives (proximité à l'océan)
### **Le pré-traitement**
* Séparation en données numériques et catégoriques
* Pipeline pour imputation des valeurs manquantes (médiane)
* Normalisation des variables numériques (StandardScaler)
* Encodage one-hot des variables catégoriques
### **La modélisation**
* Division en jeu d'entraînement et test (80%/20%), stratifiée selon une catégorie de revenu
* Construction de modèles :
  * Régression linéaire simple
  * Arbre de décision
  * Forêt aléatoire (Random Forest)
* Validation croisée 10-fold pour évaluer les performances des modèles
### Évaluation
* Calcul de l'erreur quadratique moyenne (RMSE) sur le test
* Comparaison des modèles selon leur capacité prédictive
### **La sauvegarde**
* Le meilleur modèle est enregistrer dans un fichier pickle (`rfrmodel.pkl`) l'utiliser ultérieurement.
___
## **Les résultats**
* Les modèles non linéaires (arbres de décision, forêts aléatoires) surpassent la régression linéaire simple.
* La forêt aléatoire obtient la meilleure RMSE moyenne lors de la validation croisée.
* Le pipeline assure une préparation robuste et reproductible des données.