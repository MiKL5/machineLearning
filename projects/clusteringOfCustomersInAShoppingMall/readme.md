# Segmentation de la clientèle d'un centre commercial<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat&logo=matplotlib&logoColor=white) 
![seaborn](https://img.shields.io/badge/seaborn-Statistical_Visualization-556F9F?style=flat&logo=python&logoColor=white) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div><hr>

## **Le projet**
Segmentation des clients d’un centre commercial en groupes homogènes à partir de leurs caractéristiques et comportements d’achat.  
L’objectif est de créer des clusters pour optimiser les stratégies marketing ou analyser les profils d’acheteurs.
---
## **Le dataset**
- **MallCustomers.csv** : chaque ligne correspond à un client identifié par ID avec :
  - Sexe, âge, revenu annuel (k$)
  - Score de dépenses (Spending Score 1–100)
- Prétraitement : pas de valeurs manquantes signalées.
## **La méthodologie**
### **Exploration des données**
- Visualisation du comportement d’achat (`Spending Score`) par genre.
- Scatterplots pour détecter les liens entre âge, score de dépense et revenu annuel.
### **Le préprocessing**
- Sélection et transformation des variables pertinentes (exclusion de `CustomerID`, `Gender` et `Age`).
- Préparation du tableau des features pour le clustering.
### **Le clustering**
- Utilisation de l’algorithme KMeans (scikit-learn) pour regrouper les clients.
- Détermination du nombre optimal de clusters avec la méthode du « coude » (elbow method), en traçant l’inertie totale pour différents k.
- Application finale avec 5 clusters (k=5), visualisation des centres et des groupes.
### **La visualisation**
- Diagramme de dispersion du revenu annuel et du score de dépenses, coloré par cluster.
- Affichage des centroïdes pour faciliter l’interprétation.
## **Les principaux résultats**
- Le modèle KMeans propose une segmentation claire en cinq groupes de clients, offrant une vision exploitable pour actions marketing ciblées.
- Les clients entre 20 et 40 ans présentent les scores de dépense les plus élevés.
- Le diagnostic visuel des clusters permet de comprendre les comportements selon le revenu et le score d’achat.
<!-- ## Licence
Projet distribué sous licence MIT. Voir [LICENSE](LICENSE). -->
## **NOTA**
- Pipeline reproductible, designé pour la flexibilité avec d'autres jeux de données.
- Bonnes pratiques de data science : visualisation, validation des clusters, code commenté.