# **Réduire la dimensionnalité par t-SNE et UMAP**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&logo=python&logoColor=FFD43B) 
![Scikit-learn](https://img.shields.io/badge/scikit--learn-DimReduction-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
![Plotly](https://img.shields.io/badge/Plotly-Interactive_charts-1b9e77?style=flat&logo=plotly) 
![Seaborn](https://img.shields.io/badge/Seaborn-Data_Visualization-556F9F?style=flat&logo=python&logoColor=white)

</div>

___
Ce projet illustre la **réduction de dimensionnalité** appliquée au jeu de données Iris, en utilisant les techniques t-SNE (t-distributed Stochastic Neighbor Embedding) et UMAP, avec des visualisations interactives produites via Plotly et une analyse exploratoire via Seaborn.
---
## **La méthodologie**
### **1️⃣ Préparer et explorer les données**
* Chargement du jeu de données Iris embarqué dans Plotly ou Seaborn.
* Affichage des premières lignes et sélection des features numériques (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).
### **2️⃣ La visualisation exploratoire**
* **Matrice de scatterplots (Plotly/Seaborn)** pour observer la distribution des classes réelles.
* **Pairplot** Seaborn pour explorer les relations entre variables et la séparabilité entre espèces.
### **3️⃣ La réduction de dimensionnalité**
* Application de **t-SNE** (Scikit-learn) pour projeter les données sur 2 et 3 dimensions :
```py
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=7)
projections = tsne.fit_transform(features)
```
* Possibilité d’intégrer UMAP (optionnel, nécessite l’installation de `umap-learn`).
### **4️⃣ Visualiser les projections**
* Diagramme de dispersion Plotly (2D et 3D) des données Iris réduites, coloration par `species`.
* Interactivité offerte : zoom, filtres, infobulles.
* Les clusters correspondent aux différentes espèces observées, illustrant le pouvoir séparateur de la réduction.
## **Notes**
* t-SNE et UMAP sont complémentaires : t-SNE capture les structures locales, UMAP combine local et global.
* Ces techniques sont idéales pour comprendre la structure et la séparabilité de jeux de données multi-classes.