# **Reconnaitre les chiffres manuscrits MNIST**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
![SVM](https://img.shields.io/badge/SVM-Classification-EE4C2C?style=flat&logo=scikit-learn&logoColor=white)
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div><hr>

## **La description**
Ce projet présente un pipeline complet de reconnaissance des chiffres manuscrits à partir du célèbre dataset 'MNIST'.  
Le but est de correctement classifier les images en niveaux de gris (28x28 pixels) représentant des chiffres de 0 à 9.  
Ce projet illustre les étapes clés d’un projet d’apprentissage supervisé en machine learning avec Python, scikit-learn et les techniques classiques telles que SVM et SGDClassifier.
## **Pipeline d'analyse & modélisation**
### **La préparation des données**
* Charger le dataset MNIST depuis OpenML via sklearn.datasets.
* Inspecter les données : 70 000 images 28x28 pixels vectorisées en 784 features.
* Visualiser rapidement les exemples pour comprendre la nature des données (images, distribution des labels).
* Séparer en ensembles d’entraînement (80%) et de test (20%) pour évaluer la généralisation.
### **La visualisation**
* Afficher les chiffres en images pour assurer la qualité des données.
* Analyser les performances avec un matrice de confusion, visualisée via heatmap seaborn, pour diagnostiquer les erreurs de classification.
### **La modélisation**
* Entraînement d’un classificateur SVM classique pour une reconnaissance précise (précision > 97%).
* Entraînement d’un SGDClassifier, plus rapide, optimisé via mise à l’échelle standard des données (StandardScaler).
* Evaluation des modèles avec précision et matrice de confusion.
* Comparaison des performances entre méthodes pour comprendre compromis précision/vitesse.

## **Résultats attendus et Enhancements**
Ce projet attend un taux de précision supérieur à 97% avec SVM, et environ 86-88% avec SGDClassifier après normalisation. Les matrices de confusion mettent en lumière les chiffres les plus confondus, orientant les futures améliorations.
<!-- ## **Améliorations possibles**
* Intégration de réseaux de neurones convolutionnels (CNN) pour accroitre les performances ;
* Augmenter les données pour une généralisation renforcée ;
* Crer une interface utilisateur pour tester en temps réel. -->
<!-- ___
## **Licence**
Ce projet est libre, les sources sont accessibles pour étude et reproduction. Merci de citer les sources de données et bibliothèques utilisées. -->
___
## **Remarques**
MNIST reste un excellent benchmark pour débuter la reconnaissance d’image manuscrite. Ce projet fournit une base pédagogique robuste adaptée aux novices et confirmés souhaitant comprendre les fondements pratiques du machine learning appliqué à l’image.