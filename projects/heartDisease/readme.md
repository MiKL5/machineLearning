# **Prédiction des maladies cardiaques**<a href="../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b)
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-FF6600?style=flat&logo=xgboost&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat&logo=matplotlib&logoColor=white)

</div>

## **L'objectif du projet**
Ce projet vise à prédire la présence ou l'absence de maladies cardiaques chez les patients en utilisant des algorithmes de machine learning supervisé. Il a pour but d'identifier les caractéristiques les plus déterminantes influençant le diagnostic et d’évaluer la performance des modèles prédictifs.
___
## Le jeu de données
Il contient diverses mesures cliniques et démographiques relatives aux patients, incluant :
* L'âge, sexe, tension artérielle ;
* Le cholestérol, fréquence cardiaque maximale ;
* Les résultats d'examens médicaux spécifiques ;
* L'indicateur de présence ou absence de maladie cardiaque (Heart Disease).
___
## **La méthodologie**
### **Le pré-traitement**
* Nettoyage ;
* Conversion de la colonne 'Heart Disease' en valeurs binaires ;
* Séparation des variables explicatives et de la variable 'Heart Disease'.
### **La visualisation**
* Représentation graphique des deux premières variables principales, colorée par présence ou non de problème cardiaque.
### **La modélisation**
* Évaluation de plusieurs modèles classiques :
  * Arbre de décision ;
  * Forêt aléatoire (Random Forest) ;
  * XGBoost (Gradient Boosting).
* Validation croisée 5-fold pour mesurer la robustesse des modèles ;
* Optimisation des hyperparamètres via GridSearchCV (ex : nombre d'arbres, profondeur maximale).
### **L'évaluation**
* Calcul de l'accuracy (précision) sur les jeux d'entraînement et de test ;
* Comparaison des performances pour sélectionner le meilleur modèle.
___
## **Les résultats**
* Les modèles Random Forest et XGBoost atteignent une précision proche de 89% sur le jeu de test ;
* GridSearchCV recommande une profondeur maximale de 10 pour le Random Forest ;
* Le modèle sélectionné offre un bon compromis entre complexité et performance.