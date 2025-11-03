# **Association de règles d’apprentissage avec FP-Growth**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=FFD43B) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![mlxtend](https://img.shields.io/badge/mlxtend-Frequent_Patterns-212121?style=flat&logo=python&logoColor=white)

</div><hr>

Implémentation de l’algorithme **Frequent Pattern Growth (FP-Growth)** pour découvrir les associations les plus courantes dans des transactions (listes d’items).  
La librairie utilisée est mlxtend pour faciliter la transformation et l’extraction.
---
## **Le dataset**
* Jeu de panier d'achat simulés.
* Chaque ligne correspond à une transaction (panier d’achat).
## **La méthodologie**
1. **L'encodage** : Utiliser `TransactionEncoder` pour transformer la liste des items en un tableau booléen (One-Hot Encoding).
2. **Faire le DataFrame** : Passer au format tabulaire avec pandas.
3. **Extraire des motifs fréquents** : Application de la fonction '`fpgrowth`' sur le DataFrame pour identifier les itemsets fréquents selon un seuil de support (par exemple, 60%).
   * Les itemsets sont ensuite traduits en leur valeur d’origine pour faciliter la lecture.
## **Résultat**
* Extraction des itemsets fréquents (ensembles d’items apparaissant dans au moins 60% des transactions).
* Identification des règles d’association dans les données.
## **La conclusion**
L’approche '`FP-Growth`' détecte rapidement et efficacement les associations d’items dans de grands ensembles transactionnels.  
Son usage est particulièrement adapté à l’analyse de panier d’achat, la détection de comportements récurrents ou l’exploration de motifs dans des jeux de données non étiquetés.
<!-- ## **Suggestions**
* Modifier le seuil de support pour ajuster la sensibilité à la fréquence des motifs.
* Tester sur des datasets réels (panier supermarché, logs utilisateurs…).
* Compléter avec l’extraction de règles d’association (`apriori`, `association_rules`). -->