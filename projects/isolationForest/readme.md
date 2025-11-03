# Détection d'anomalies avec Isolation Forest en Python <a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=FFD43B) 
![Scikit-learn](https://img.shields.io/badge/scikit--learn-AnomalyDetection-FF6600?style=flat&logo=scikit-learn&logoColor=white) 
![Seaborn](https://img.shields.io/badge/Seaborn-Data_Visualization-556F9F?style=flat&logo=python&logoColor=white)

</div><hr>

Ce projet illustre l'utilisation de la méthode **Isolation Forest** pour détecter les anomalies (valeurs aberrantes) dans un jeu de données simple.  
Le script utilise un petit dataset simulé où la variable d'intérêt est la note (`Marks`) d'étudiants.
---
## **Le dataset**
* Contient 15 observations avec 2 colonnes :  
  * `StudentId` : identifiant unique  
  * `Marks` : notes des étudiants (variable à analyser)
## **La méthodologie**
1. Visualisation initiale des données avec un **boxplot** pour repérer visuellement les valeurs aberrantes.  
2. Création d'un modèle **Isolation Forest** avec un taux de contamination fixé à 1% (proportion attendue d'anomalies).  
3. Entraînement du modèle sur la variable `Marks`.  
4. Prédiction des valeurs aberrantes avec le modèle.  
5. Ajout de la colonne `Outlier` dans le DataFrame indiquant pour chaque observation si elle est considérée anormale (`-1`) ou normale (`1`).
## **Le résultat**
Le modèle identifie l'observation avec `StudentId = 10` comme valeur aberrante.
## **La conclusion**
L’**Isolation Forest** est une méthode efficace et rapide pour détecter des anomalies dans des jeux de données univariés ou multivariés.  
Cette approche est particulièrement utile lorsque les données ne sont pas étiquetées et que la proportion d’anomalies est faible.