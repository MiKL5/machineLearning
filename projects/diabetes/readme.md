# **Prédire le diabète**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">


![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white)

</div>

## **L'objectif**
L'intérêt est de prédire la présence ou l'absence du diabète chez les patients à partir de données cliniques et démographiques.  
L'objectif est d'identifier les caractéristiques les plus significatives pour le diagnostic, de comparer plusieurs modèles de machine learning classiques, puis déterminer le modèle offrant les meilleures performances.
---
## **Le jeu de données**
Le dataset contient ces mesures cliniques et démographiques
* Nombre de grossesses, glucose, pression artérielle ;
* Épaisseur du pli cutané, insuline ;
* Indice de masse corporelle (IMC), fonction pedigree du diabète ;  
* Âge du patient ;
* Indicateur binaire de présence ou absence du diabète.
---
## **La méthodologie**
### **Le pré-traitement des données**
* Nettoyer et vérifier les valeurs manquantes
* Séparer les variables explicatives et de la cible (diabète ou non)
### **La visualisation**
* Histogrammes et distributions des variables
* Corrélations entre variables clés
### **La modélisation**
* Évaluation de plusieurs modèles de classification supervisée
  * Arbre de décision
  - Forêt aléatoire (Random Forest)
* Optimisation des hyperparamètres via GridSearchCV (ex. profondeur maximale, nombre d’arbres)
### **L'évaluation**
* Mesure de la précision (accuracy) sur les jeux d’entraînement et test
* Comparaison des performances pour sélectionner le meilleur modèle
## **Les principaux résultats**
* La forêt aléatoire offre généralement une meilleure précision que l’arbre de décision seul
* L’optimisation par `GridSearchCV` améliore la robustesse du modèle