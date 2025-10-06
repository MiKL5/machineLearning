# **Prédire la maladie de Parkinson**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
![SVM](https://img.shields.io/badge/SVM-Classification-EE4C2C?style=flat&logo=scikit-learn&logoColor=white)
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div><hr>

## **L'objectif**
Détecter la maladie de Parkinson à partir de données cliniques grâce à l'apprentissage automatique. À partir de paramètres mesurés sur la voix, il permet de prédire la présence ou non de la maladie à l’aide de modèles de classification, avec un accent particulier sur le Support Vector Machine (SVM).
---
## **Le jeu de données**
Le jeu de données utilisé contient des mesures biomédicales extraites de signaux vocaux, avec des variables telles que :
* Mesures dérivées de la fréquence fondamentale (MDVP) ;
* Indicateurs de perturbations (Jitter, Shimmer) ;
* Rapport bruit-harmonique (NHR, HNR) ;
* Statut de santé (`status` : 1 = Parkinson, 0 = sain).

> **N. B.** Le dataset a été fourni à un but académique par un enseignant. Il est primordial de vérifier les droits associés avant toute utilisation ou diffusion.
___
## **La méthodologie**
### **Le prétraitement**
* Analyse structurelle et des valeurs manquantes ;
* Suppression des colonnes non-informatives (`name`) ;
* Séparation des features explicatives et du label (`status`) ;
* Normalisation par `StandardScaler` des valeurs numériques.
### **La modélisation**
* Split en ensembles d’entraînement (80%) et de test (20%, stratifié) ;
* Entraînement d’un modèle SVM pour classification binaire ;
* Évaluation de la précision (`accuracy`) sur le test set.
### **L'évaluation**
* Affichage des scores de classification et exploration des performances.
## **Les principaux résultats**
* Le SVM atteint généralement une précision élevée sur le jeu de test ;
* Les étapes de standardisation et de sélection des features sont importantes pour la réussite de la modélisation.
