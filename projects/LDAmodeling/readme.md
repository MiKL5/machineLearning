# Modélisation thématique de documents avec LDA (Latent Dirichlet Allocation)<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&logo=python&logoColor=FFD43B) 
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-Topic_Modeling-F7931E?style=flat&logo=scikit-learn&logoColor=white)

</div><hr>

## **Le projet**
Démontrer la **modélisation thématique (topic modeling)** par l’algorithme LDA (*Latent Dirichlet Allocation*).  
L’algorithme attribue à chaque document une probabilité d’appartenance à différents sujets (“topics”), facilitant l’organisation ou l’analyse exploratoire de grandes collections de textes.
## **La méthodologie**
### **1️⃣ Préparation et vectorisation des textes**
* Les documents exemples abordent plusieurs thèmes, principalement *Technologie* et *Sports*.
* Textes transformés en matrice numérique “sac de mots” (binarisation, sélection des mots fréquents) via `CountVectorizer` :
```py
vectorizer = CountVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(documents)
```
### **2️⃣ La modélisation thématique**
* Application de LDA avec 2 topics sous scikit-learn :
```py
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)
```
* Extraction pour chaque document du *topic* ayant la plus forte probabilité d’appartenance (`lda.transform(X).argmax(axis=1)`).
* Regroupement des documents par thème dominant.
### **3️⃣ Exploration des résultats**
* Affichage des groupes de documents organisés par sujets détectés :
  * Remise de chaque texte dans son groupe thématique principal.
  * Interprétation directe :
    * *Technology* : sujets liés à l’innovation, intelligence artificielle, réalité virtuelle…
    * *Sports* : sujets concernant athlètes, événements et bienfaits du sport.
## Notes
* Pour un corpus volumineux, ajuster le nombre de topics (`n_components`) et le prétraitement des textes pour de meilleurs résultats.
* 'LDA' est un modèle probabiliste, chaque document peut avoir une part dans plusieurs topics, mais est ici classé selon le plus probable.