# **Clustering non supervisé sur le jeu de données Iris**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat&logo=matplotlib&logoColor=white) 
![Seaborn](https://img.shields.io/badge/seaborn-Statistical_Visualization-556F9F?style=flat&logo=python&logoColor=white) 
![Plotly](https://img.shields.io/badge/plotly-Interactive_Visualizations-FF69B4?style=flat&logo=plotly&logoColor=white) 
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div><hr>

## **Le projet**
Appliquer des méthodes d’apprentissage non supervisé pour identifier des groupes naturels dans le célèbre jeu de données Iris.  
L’objectif est de segmenter les données sans utiliser d’étiquettes, en apportant une compréhension approfondie via plusieurs algorithmes de clustering et des visualisations riches.
---
## Le jeu de données
Le dataset Iris de `sklearn.datasets` contient 150 échantillons avec 4 caractéristiques continues :  
* Longueur et largeur du sépal.
* Longueur et largeur du pétale.

Pas d’étiquettes utilisées pour entraîner les modèles, conformément à l’apprentissage non supervisé.
## **Méthodologie**
### **Prétraitement**
* Standardisation des features avec `StandardScaler` pour homogénéiser les échelles.
### **Clustering et Modèles**
* `KMeans` clustering avec recherche du nombre optimal de clusters via la **méthode du coude** et calcul du **silhouette score**.
* `DBSCAN` pour clustering basé sur la densité sans besoin de définir le nombre de clusters.
* Le clustering hiérarchique agglomératif (`AgglomerativeClustering`) avec visualisation par dendrogramme.
### **Les visualisations**
* Scatterplots 2D et 3D colorés par cluster (avec Plotly pour interactivité).
* Pairplot pour visualiser les relations entre les caractéristiques dans chaque cluster.
* Affichage des centroïdes pour KMeans.
* Dendrogramme illustrant la hiérarchie dans le clustering agglomératif.
### **L'évaluation**
* **Silhouette score** : mesure de cohésion et séparation des clusters (valeurs entre -1 et 1).
* **Adjusted Rand Index (ARI)** : comparaison des clusters avec les vraies espèces Iris (informations hors entraînement, à titre d’analyse).
## **Les résultats clés**
* `KMeans` avec trois clusters révèle une segmentation pertinente correspondante aux espèces réelles.
* `DBSCAN` permet d’identifier des groupes selon la densité locale des points, intéressant pour détacher des outliers.
* Les clustering hiérarchique propose une vision différente de la structure de regroupement des données.
* Des visualisations riches et interactives facilitent l’interprétation des clusters.
<!-- ## Licence
Projet sous licence MIT - voir le fichier [LICENSE](LICENSE). -->
___
## **Note<!--s-->**
* L'approche 100 % non supervisée, sans utilisation de labels pour apprentissage.
<!-- * Idéal pour initiation au clustering avec les outils Python standards. -->
<!-- * Il y a des visualisations avancées pour faciliter l’interprétation. -->