# Détection d’anomalies avec One-Class SVM<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=FFD43B) 
![Scikit-learn](https://img.shields.io/badge/scikit--learn-AnomalyDetection-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
![Seaborn](https://img.shields.io/badge/Seaborn-Data_Visualization-556F9F?style=flat&logo=python&logoColor=white)

</div><hr>

Ce projet démontre comment utiliser **One-Class SVM** pour détecter des **valeurs aberrantes (outliers)** dans un jeu de données non étiqueté.  
Cette méthode est adaptée à l’**apprentissage non supervisé**, et vise à modéliser la frontière des observations normales afin d’identifier les anomalies.
---
## **Le dataset**
Les deux variables du dataset **Iris** utilisées pour la détection :
* `sepal_width`
* `petal_width`
## **La méthodologie**
### **1️⃣ Chargement et exploration des données**
- Import du jeu Iris directement depuis Seaborn.
- Observation des variables : largeurs du sépale et du pétale.
### **2️⃣ Application du modèle One-Class SVM**
Lors de la construiction du modèle, le paramètre `nu` contrôle la proportion maximale de données considérées comme anomalies.  
Plus `nu` est élevé, plus le modèle est permissif à la présence d’outliers.

À l'entraînement les valeurs sont :
* `1` pour les observations normales  
* `-1` pour les anomalies
### **3️⃣ La visualisation**
Lors du représentation graphique, les outliers apparaissent visuellement distincts des points majoritaires.
## **Les principaux résultats**
* One-Class SVM identifie efficacement les observations atypiques du dataset Iris.  
* Les points détectés comme anomalies peuvent correspondre à des spécimens inhabituels ou à du bruit dans la mesure.
## **NOTA**
* One-Class SVM est idéal pour les jeux de données sans labels où seule la détection de points atypiques est requise.  
* Pour des ensembles volumineux, il peut être avantageusement remplacé par **Isolation Forest** ou **Local Outlier Factor**.