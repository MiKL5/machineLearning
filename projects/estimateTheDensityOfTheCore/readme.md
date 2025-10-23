# **Estimer de la densité de noyau (KDE) avec Seaborn et SciPy**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Seaborn](https://img.shields.io/badge/seaborn-Data_Visualization-556F9F?style=flat&logo=python&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/matplotlib-Plotting-11557C?style=flat&logo=matplotlib&logoColor=white) 
![NumPy](https://img.shields.io/badge/numpy-Numerical_Computing-013243?style=flat&logo=numpy) 
![SciPy](https://img.shields.io/badge/scipy-Scientific_Computing-8CAAE6?style=flat&logo=scipy) 
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div><hr>

Estimer la **densité de probabilité** sous-jacente à un jeu de données à l’aide de la méthode non paramétrique dite de **l’estimation par noyau** (Kernel Density Estimation - KDE).  
Il utilise la fonction `kdeplot` de Seaborn pour des courbes univariées et bivariées, et `gaussian_kde` de SciPy pour une représentation plus avancée en 3D.
---
## **Les données utilisées**
* Un tableau de nombres entiers simulant des observations (échantillon).
* Le jeu de données `mpg` (mileage per gallon) issu de Seaborn, contenant des caractéristiques automobiles classiques.
## **Fonctionnalités et méthodes**
### **1️⃣ La visualisation univariée (Histogramme + KDE)**
* Un histogramme simple avec `sns.histplot`.
* KDE univariée ajustée avec différents paramètres `bw_method` pour contrôler la largeur de la bande (lissage).
### **2️⃣ La gestion des données manquantes**
* Supprmer les lignes aux données manquantes dans la colonne `horsepower` avant analyse.
### **3️⃣ La visualisation bivariée**
* KDE bivariée sur deux variables, par exemple consommation (`mpg`) et puissance (`horsepower`), avec ou sans courbes de niveaux.
*Visualisation en 3D de la densité selon les deux variables grâce à une grille de points et `gaussian_kde`.
### **4️⃣ L'analyse par catégories**
* Visualisation multi-catégories par exemple selon le nombre de cylindres.
*Variations de densité en fonction de catégories, visualisation avec couleurs et barres colorées pour faciliter la compréhension.
## **Les résultats**
* L 'estimations précises des densités sous-jacentes pour données simples et complexes.
* La visualisation intuitive par courbes KDE lissées.
* La représentation graphique en 3D pour observer la concentration de points dans l’espace des features.
## **Notes**
* L’estimation KDE est une méthode puissante pour comprendre la distribution des données sans hypothèses paramétriques.
* Le paramètre `bw_method` est clé pour ajuster le compromis biais-variance.
* Les visualisations multiples (1D, 2D, 3D) permettent une meilleure interprétation.