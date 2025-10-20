# **L'algorithme "Apriori"**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>

<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Processing-150458?style=flat&logo=pandas&logoColor=white) 
![MLxtend](https://img.shields.io/badge/mlxtend-Apriori_Algorithm-008080?style=flat&logo=python&logoColor=white) 
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div><hr>

## **Le projet**
Illustre l’utilisation de l’algorithme **Apriori** pour identifier les associations fréquentes entre produits dans un ensemble de transactions.  
L’objectif est d’effectuer une **analyse de panier** (*Market Basket Analysis*) afin de découvrir des modèles de co-achat — essentiels pour les stratégies commerciales (ex. promotions, agencement de produits, recommandations).

Chaque ligne représente un panier client contenant plusieurs produits.
## **La méthodologie**
### **1️⃣ La préparation des données**
* Création d’un **DataFrame Pandas** à partir d’une liste de transactions.
* Transformation de la colonne *Items* en chaîne de caractères.
* Application d’un **encodage one-hot (binaire)** via `str.get_dummies(sep=',')`, convertissant chaque transaction en un vecteur de 0 et 1.
### **2️⃣ L'extraction des ensembles fréquents**
* Application de l’algorithme **Apriori** pour calculer les combinaisons d’articles apparaissant fréquemment dans les paniers :
frequent_itemsets = apriori(onehot, min_support=0.4, use_colnames=True)
* Le paramètre `min_support=0.4` indique que seuls les ensembles d’articles présents dans au moins **40 % des transactions** seront retenus.
### **3️⃣ Générer les règles d'association**
* Extraction des **règles d’association** avec la fonction `association_rules` :
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
* Seules les règles présentant une **confiance ≥ 0,7 (70 %)** sont conservées.
### **4️⃣ Analyser les résultats**
* Les principales métriques calculées :
* **Support** : fréquence de l’ensemble d’articles dans les transactions.  
* **Confidence** : probabilité d’achat du produit B lorsque le produit A est acheté.  
* **Lift** : degré de dépendance (valeur > 1 ⇒ corrélation positive).
## **Exemple de résultats**
Règle | Support | Confiance | Lift
---|---|---|---
pain → lait | 0.6 | 0.8 | 1.25
lait → couche | 0.4 | 0.75 | 1.40

Si un client achète du **pain**, la probabilité qu’il achète également du **lait** est de **80 %**.