# **Segmentater la clientèle par le clustering hiérarchique**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![Scipy](https://img.shields.io/badge/scipy-Hierarchical_Clustering-8CAAE6?style=flat&logo=scipy) 
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat&logo=matplotlib&logoColor=white)

</div><hr>

Ce projet applique un **clustering hiérarchique agglomératif** sur un jeu de données clients d’un centre commercial.  
L’objectif est de segmenter la clientèle en groupes homogènes pour mieux cibler les actions commerciales et marketing.
---
## **La méthodologie**
### **1. Préparer les données**
* Sélection des colonnes pertinentes (`Annual Income (k$)` et `Spending Score (1-100)`).
### **2. L'analyse hiérarchique par dendrogramme**
* Construction d’un dendrogramme avec la méthode de Ward via `scipy.cluster.hierarchy.linkage`.  
* Visualisation du dendrogramme pour déterminer le nombre optimal de clusters.  
* Tracé d’une ligne horizontale sur le dendrogramme pour illustrer la coupure des clusters.
### **3. Le clustering agglomératif**
* Créer un `AgglomerativeClustering` avec le nombre de clusters voulu.  
* Ajustement aux données et prédiction des labels cluster.
### **4. Visualiser des clusters**
* Scatter plot des points clients colorés selon leur cluster.  
* L'analyse qualitative des groupes :  
  * Les revenus élevés versus faibles.  
  * Les scores de dépenses élevés vs faibles.  
* L'interprétation marketing.
## **Notes**
* Le dendrogramme est un outil puissant pour choisir le nombre de clusters.
* L’approche hiérarchique permet une exploration progressive des regroupements.  
* Idéal pour des bases clients aux comportements diversifiés comme les centres commerciaux.