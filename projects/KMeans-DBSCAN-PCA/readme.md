# Analyse et clustering non supervisé : KMeans, DBSCAN et PCA<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering%20%26%20PCA-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat&logo=matplotlib&logoColor=white) 
![Seaborn](https://img.shields.io/badge/seaborn-Statistical_Plots-556F9F?style=flat&logo=python&logoColor=white) 
![OpenML](https://img.shields.io/badge/OpenML-Dataset_Integration-000000?style=flat) 
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div><hr>

## **Le projet**
Ce projet combine plusieurs techniques d’**apprentissage non supervisé** – **KMeans**, **DBSCAN**, et **PCA** – pour explorer et visualiser la structure interne de jeux de données classiques en data science (Iris & MNIST).  
L’objectif est de segmenter, analyser et réduire la dimensionnalité des données pour interpréter les regroupements visuels.
---
## Jeux de données utilisés
### Dataset **Iris**
* Disponible via `seaborn.load_dataset("iris")`  
* 150 échantillons de fleurs avec 4 caractéristiques : longueur/largeur de sépale et pétale.  
* Les algorithmes **KMeans** et **DBSCAN** y sont appliqués pour découvrir des regroupements naturels entre :  
  *Setosa*, *Versicolor* et *Virginica*.
### **Dataset '`MNIST`'**
* Importé via `fetch_openml('mnist_784', version=1)`  
* Contient 70 000 images 28×28 de chiffres manuscrits.  
* Utilisé ici pour démontrer la **Réduction de dimensionnalité (PCA)** à 2 composantes principales afin de visualiser la distribution de chiffres dans un espace réduit.
---
## Méthodologie
### **1️⃣ Visualisation initiale (Iris)**
* Graphiques en nuages de points (`seaborn.scatterplot`) entre paires de variables pour explorer les distributions naturelles.
* Coloration par espèce réelle pour observation de la structure des groupes.
### **2️⃣ Clustering KMeans**
* Application du modèle `KMeans(n_clusters=3)`.
* Entraînement sur deux dimensions : *longueur du sépale* et *longueur du pétale*.
* Visualisation colorée selon le cluster prédit.
* Observation d’un bon regroupement général malgré certains points ambiguës aux frontières.
### **3️⃣ Clustering DBSCAN**
* Utilisation d’un modèle `DBSCAN(eps=1, min_samples=6)`.
* Détection automatique des groupes denses et des points isolés (*outliers*).
* Visualisation contrastée en 2D : certains clusters fusionnent ou se réduisent, illustrant la robustesse de la densité locale.
### **4️⃣ Réduction dimensionnelle PCA sur MNIST**
* Normalisation des données (`/255.0`) pour passage en échelle [0, 1].  
* Calcul avec `PCA(n_components=2)` pour projeter les 784 features en un espace bidimensionnel.
* Visualisation du résultat en scatter colorisé selon les labels réels (à but illustratif).
## **Les résultats principaux**
* **KMeans** : reproduction efficace de la structure observée sur Iris (3 clusters cohérents).  
* **DBSCAN** : détection des zones de haute densité — plus sélectif que KMeans, utile sur données bruitées.  
* **PCA sur MNIST** : réduction globale à 2D maintenant la variabilité principale, observation claire de certains regroupements numériques.
<!-- ## **Licence**
Ce projet est publié sous licence **MIT**.  
Vous pouvez le réutiliser, le modifier et le redistribuer librement en citant la source. -->
___
## **Notes**
- Approche entièrement **non supervisée** (aucune variable cible utilisée pour apprentissage).  
- KMeans = segmentation « globale », DBSCAN = segmentation « locale ».  
- PCA = réduction de dimension pour interprétation humaine.  
- Idéal pour la démonstration de clustering hybride et visualisation multi-datasets.