# Partitionnement de données avec un modèle Bayésien Gaussien (Bayesian Gaussian Mixture)<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=FFD43B)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering-FF9900?style=flat&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Data_Visualization-556F9F?style=flat&logo=python&logoColor=white)

</div><hr>

Illustrer l'utilisation du modèle de mélange gaussien bayésien (Bayesian Gaussian Mixture Model) pour réaliser un partitionnement non supervisé des données du célèbre dataset Iris.  
L'objectif est de segmenter les observations en clusters correspondant aux différentes espèces d'iris sans utiliser les labels.
---
## **Le dataset**
* Dataset Iris de Seaborn ➜ longueur et largeur des pétales et sépales de trois espèces d'iris.  
* Variables utilisées ➜ toutes les variables numériques sauf la colonne `species` contenant le label.
## **La méthodologie**
1. Visualiser avec un scatterplot des variables `petal_length` et `sepal_length` colorié par espèce réelle.  
2. Créer un estimateur `BayesianGaussianMixture` avec trois composantes (clusters) supposées.  
3. Ajuster (`fit`) du modèle sur les variables explicatives `X`.  
4. Prédire des clusters pour chaque observation.  
5. Visualiser des clusters obtenus sur les mêmes variables `petal_length` et `sepal_length` avec un nuage de points colorié en fonction du cluster assigné.
## **Les résultats**
* Le modèle identifie trois clusters distincts qui correspondent assez bien aux espèces réelles.
* Le modèle bayésien offre l’avantage d’incorporer une régularisation qui évite le surapprentissage et ajuste automatiquement la complexité du modèle.
## **Conclusion**
Le modèle Bayesian Gaussian Mixture est adapté pour l’analyse non supervisée et la segmentation de données multivariées continues.  
Sa flexibilité permet l’adaptation automatique du nombre de clusters et une meilleure généralisation par rapport aux mélanges gaussiens classiques.
<!-- ## Suggestions
* Ajuster les paramètres '`n_components`' et '`covariance_type`' pour améliorer la segmentation selon les données.
* Comparer avec d’autres méthodes de clustering comme K-Means ou DBSCAN.
* Utiliser ce modèle comme prétraitement pour d’autres tâches de classification ou de détection d’anomalies. -->