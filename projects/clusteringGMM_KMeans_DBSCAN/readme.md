# Clustering comparatif avec GMM, K-Means et DBSCAN<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&logo=python&logoColor=FFD43B) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
![Seaborn](https://img.shields.io/badge/seaborn-Statistical_Visualization-556F9F?style=flat&logo=python&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/matplotlib-Plotting-11557C?style=flat&logo=matplotlib&logoColor=white) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white)

</div><hr>

Ce projet explore et compare trois algorithmes de **clustering non supervisé**  
1️⃣ **K-Means** pour ségmenter par centroïdes.  
2️⃣ **Gaussian Mixture Model (GMM)** pour modèliser les clusters de façon probabiliste.  
3️⃣ **DBSCAN** pour regrouper en se basant sur la densité des données.
---
Deux jeux de données sont utilisés  
1️⃣ **Iris dataset** (via Seaborn) — pour comparer KMeans et GMM dans un contexte réel.  
2️⃣ **Moons dataset** (via scikit-learn) — pour visualiser un cas de données non linéairement séparables où DBSCAN excelle.
---
## **La méthodologie**
### **1️⃣ Le chargement des données**
* **Jeu Iris** : importé via `sns.load_dataset("iris")`.
* **Jeu Moons** : généré par `make_moons(200, noise=0.05)`.
### **2️⃣ La préparation**
* Suppression des labels (`species`) pour ne garder que les features numériques.
* Réduction de dimension à 2 composantes via **PCA**.
* Conversion en DataFrame pour visualisation Seaborn.
### **3️⃣ Les algorithmes appliqués sont**
#### **Gaussian Mixture Model (GMM)**
Il permet une segmentation basée sur des distributions de probabilité.  
À l'implémentation, GMM peut estimer les densités et détecter des anomalies par probabilité d’appartenance.
#### **K-Means**
C'est un algorithme de référence pour le clustering simple.  
Il est appliqué sur les données Moons et est inefficace sur des structures non linéaires.
#### **DBSCAN**
La segmentation basée sur la densité, efficace pour les formes complexes et le bruit.
À l'implémentation, la capacité à détecter les outliers et à s’adapter à des formes irrégulières.
---
## Visualisations
1. **Scatter plots Seaborn** pour le dataset Iris (comparaison entre vrais labels et clusters prédits).
2. **Matplotlib plots** pour le dataset Moons (K-Means vs DBSCAN).
Les figures montrent que :
* GMM ≈ K-Means sur Iris (clusters elliptiques bien séparés).
* DBSCAN surperforme K-Means sur des structures non convexes.
___
## **NOTA**
* GMM permet de modéliser des clusters elliptiques avec incertitude probabiliste.
* DBSCAN détecte les structures complexes sans paramètre de cluster prédéfini.
* Utiliser **PCA** pour simplifier les jeux multidimensionnels avant visualisation.