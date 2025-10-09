# Prédiction de la performance étudiante<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b) 
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
<!-- ![MIT License](https://img.shields.io/badge/License-MIT-blue.svg) -->

</div><hr>

## **Le projet**
Prédire l’indice de performance des étudiants à partir de leurs données personnelles et d’activité extrascolaire.  
Le pipeline comprend un pré-traitement des données catégorielles, la division en ensembles d’entrainement et de test, puis la modélisation par plusieurs algorithmes de régression supervisée.
## **Les jeu de données**
Le dataset `Student_Performance.csv` contient des données quantitatives et qualitatives relatives aux étudiants, avec comme variable cible :  
* **Performance Index** : score de performance à prédire.  
* **Extracurricular Activities** : variable catégorielle encodée par `One-Hot Encoding`.
## **La méthodologie**
### **Le pré-traitement**
* Chargement et exploration initiale des données : formes, iformations,Diviser manquantes.  
* Encodage One-Hot * la variable catégorielle `Extracurricular Ativities`Diviser
* Séparation des *atres  et du label (`y`).  
### **Diviser des données**
* Découpage aléatore s d’entraînement (80%) et test (20%) avec `random_state=7` pour la reproductibilité.
### **Modééliser et évaluer**
* Support Vector Regressor (SVR)  
* Régression Linéaire standard  
* Régression Linéaire avec pipeline de normalisation des features (`StandardScaler`)  
* Utilisation de la métrique **Mean Absolute Error (MAE)** pour mesure des erreurs de prédiction.
<!-- ___
## **Licence**
Ce projet est publié sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
## Notes finales
Ce projet met en avant les meilleures pratiques en data science : documentation claire, pipeline reproductible, usage judicieux des techniques d’encodage et de scaling, et évaluation rigoureuse par métriques adaptées.  
Idéal pour un cadre académique ou professionnel. -->