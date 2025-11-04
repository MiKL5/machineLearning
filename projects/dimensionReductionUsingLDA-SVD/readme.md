# Réduire la dimensionnalité avec LDA et SVD tronquée<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=FFD43B)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Dimensionality_Reduction-FF9900?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=flat&logo=matplotlib&logoColor=white)

</div><hr>

Ce projet montre comment appliquer deux techniques classiques de réduction de dimensionnalité sur le dataset Iris en Python :  
* **Linear Discriminant Analysis (LDA)**, une méthode supervisée qui maximise la séparation entre classes,  
* **Truncated Singular Value Decomposition (SVD)**, une méthode non supervisée similaire à la PCA.
---
## **Le dataset**
* Dataset Iris : quatre variables mesurent les caractéristiques florales et un label catégoriel `species`.
## **La méthodologie**
1. Séparation des données en ensembles d’entraînement et test (80% / 20%).  
2. Normalisation des données avec `StandardScaler` (important pour SVD).  
3. Application de LDA sur les données d’entraînement avec leurs labels (`y_train`).  
4. Application de SVD tronquée sur les données d’entraînement sans utiliser les labels.  
5. Visualisation des deux réductions en deux dimensions avec un nuage de points colorié selon la classe réelle.
## **Les isualisations**
* Le graphique LDA montre des clusters bien séparés correspondant aux espèces.
* Le graphique SVD révèle la structure principale de variance sans tenir compte des classes.
## **Conclusion**
* LDA est plus adapté lorsqu'on connaît les classes et souhaite maximiser leur séparation.
* La SVD est utile pour explorer la structure globale des données sans labels.
* Ces méthodes permettent d’alléger la complexité des données tout en préservant l’information essentielle.
<!-- ## Suggestions
* Expérimenter avec différentes composantes et techniques de réduction (PCA, t-SNE, UMAP).  
* Utiliser ces réductions pour améliorer des modèles de classification ou de clustering. -->