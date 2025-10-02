# **Système de recommandation de films**<a href="../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
<div align="center">

![Python](https://img.shields.io/badge/python-3.13-blue?style=flat&logo=python&logoColor=ffd43b)
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-Scientific_Computing-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat&logo=matplotlib&logoColor=white)

</div>

## **L'objectif**
Ce projet implémente un **système de recommandation de films** basé sur le contenu (*Content-Based Filtering*).  
Il utilise les métadonnées des films (genres, mots-clés, tagline, casting et réalisateur) pour calculer les similarités et proposer des suggestions pertinentes.
## **Le fonctionnalités**
* Chargement et prétraitement des données depuis `movies.csv`
* Nettoyage des valeurs manquantes
* Combinaison des caractéristiques textuelles
* Vectorisation avec **TF-IDF**
* Calcul de la similarité cosinus
* Recherche des films similaires à partir d’une saisie utilisateur
* Recommandation des **30 films les plus proches**
___
## **La méthodologie**
### **Les données**
Le script attend un fichier `movies.csv` contenant au minimum ces colonnes : `index`, `title`, `genres`, `keywords`, `tagline`, `cast`, `director`.
### **Le prétraitement**
1. Sélection des colonnes pertinentes.
2. Remplacement des valeurs manquantes par une chaîne vide.
3. Concaténation `genres + keywords + tagline + cast + director` pour chaque film.
### **La représentation**
* TF‑IDF via `sklearn.feature_extraction.text.TfidfVectorizer()` sur le texte combiné.
* Vecteurs normalisés.
### **La similarité**
* Similarité cosinus entre tous les vecteurs via `sklearn.metrics.pairwise.cosine_similarity()`.
* Matrice dense `N x N` où `N` est le nombre de films. Complexité mémoire O(N²).
### **La recherche et les recommandation**
* L’utilisateur saisit un titre.
* `difflib.get_close_matches()` retrouve le titre le plus proche.
* Le système récupère l’index du film et trie les scores de similarité décroissants.
* Retour des premiers `k` titres exclus le film source.
<!-- ### Les limitations connues
* Dépend de la qualité et de la complétude des métadonnées.
* Sensible aux variantes lexicales non normalisées.
* Cold‑start pour titres sans métadonnées.
* Scalabilité limitée par la matrice dense.
### **Les améliorations possibles**
* Lemmatisation et suppression de mots vides avec spaCy.
* Pondération différenciée des champs (par exemple plus de poids pour `director` et `cast`).
* Passage à des embeddings sémantiques (SentenceTransformers) pour capter le sens.
* Indexation approximative (FAISS, Annoy) pour grande échelle.
* Approche hybride combinant filtrage collaboratif et filtrage par contenu. -->
___
## **Les résultats**
Le script renvoie une liste ordonnée de recommandations. Exemple d’exécution :
```
Entre ton film préféré pour obtenir des recommandations similaires : Iron Man 2

Films pouvant t'intéresser :

1 . Iron Man
2 . Avengers
3 . Thor
4 . Captain America: The First Avenger
...
```
### **Protocole d’évaluation recommandé**
1. Constituer un jeu de test à partir d’un jeu complet. Par exemple retirer aléatoirement une portion des métadonnées ou masquer des associations réelles.
2. Pour chaque film de test, exiger que le système retrouve des titres pertinents.
3. Calculer les métriques suivantes :
   * **Precision@k** : la proportion d’éléments pertinents dans les k premiers résultats.
   * **Recall@k** : la proportion d’éléments pertinents retrouvés parmi tous les éléments pertinents.
   * **MAP@k** : la moyenne de la précision sur les positions des éléments pertinents.
   * **nDCG@k** : la qualité du classement prenant en compte la position.
### **Extraits de code pour Precision@k**
```py
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = sum(1 for r in recommended_k if r in relevant)
    return hits / k
```
### **L'interprétation**
* Les valeurs élevées de **Precision@k** indiquent des recommandations immédiatement pertinentes.
* Le nDCG pénalise les éléments pertinents mal classés.
* Les métriques dépendent du jeu d’évaluation. Utiliser plusieurs jeux pour robustesse.