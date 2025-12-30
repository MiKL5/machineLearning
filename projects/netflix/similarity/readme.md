# 🔎 Analyse de Similarité des contenus de <a href="#"><img align="center" src="../../../assets/netflix.png" alt="netflix" height="36px"></a>etflix<a href="../../../"><img align="right" src="https://github.com/MiKL5/Python/blob/master/assets/logo/Jupyter.svg" alt="Jupyter" height="64px"></a>
## 📌 Contexte
>Ce module complète le système de recommandation en explorant les **relations de similarité entre contenus**.  
À travers plusieurs visualisations (carte thermique, graphe de proximité, projection t-SNE), il met en évidence les **structures internes du catalogue** (films, séries) et permet de mieux comprendre la logique des recommandations.
---
## **🎯 Objectif**
> Démontrer les compétences en **Machine Learning, NLP, Data Science et Visualisation**
---
## ⚙️ Fonctionnalités principales
- **Carte thermique (heatmap)**  
  - Visualisation de la matrice de similarité cosinus pour un sous-ensemble de titres.  
  - Valeurs proches de `1` → contenus similaires.  
  - Valeurs proches de `0` → contenus dissemblables.  

- **Graphe de similarité**  
  - Construction d’un graphe basé sur les similarités cosinus supérieures à un seuil.  
  - Les **nœuds** représentent des titres.  
  - Les **arêtes** représentent une proximité de genres.  
  - Mise en évidence de **clusters thématiques** (horreur, comédie, drame, etc.).  

- **Projection t-SNE (t-distributed Stochastic Neighbor Embedding)**  
  - Réduction de dimensions des vecteurs TF-IDF en 2D.  
  - Chaque point correspond à un film ou une série.  
  - Les clusters visuels représentent des regroupements thématiques.  
  - Identification d’**outliers** : contenus atypiques ou très spécifiques.  
## 🛠️ Stack technique
- **Langage** : Python 3.x  
- **Bibliothèques principales** :
  - `pandas` → manipulation des données  
  - `scikit-learn` → TF-IDF, t-SNE  
  - `matplotlib` → visualisations  
  - `seaborn` → heatmap  
  - `networkx` → graphe de similarité  
## 🚀 Exemple d’utilisation
### 📊 Heatmap
```python
sns.heatmap(cosine_sim_subset, xticklabels=titles_subset, yticklabels=titles_subset)
plt.title('Heatmap de similarité cosinus (TF-IDF genres)')
```
### 🔗 Graphe
```python
nx.draw(G, pos, with_labels=True, node_size=700, edge_color='orange', node_color='skyblue')
plt.title("Graphe de similarité cosinus des titres")
```
### 🌐 Projection t-SNE
```python
tsne = TSNE(n_components=2, metric='cosine', random_state=42)
embedding = tsne.fit_transform(tfidf_matrix.toarray())
plt.scatter(embedding[:,0], embedding[:,1])
```
## 📊 Résultats
- **Carte thermique** : montre les similarités fortes/faibles entre titres.  
- **Graphe** : met en évidence des **communautés de contenus** (clusters).  
- **t-SNE** : offre une **vue d’ensemble intuitive** du catalogue et révèle des regroupements naturels.  
<!-- 
---
## 🌍 Perspectives d’évolution
- Ajustement dynamique du **seuil de similarité** pour affiner le graphe.  
- Comparaison entre **genres, pays, années** pour enrichir les visualisations.  
- Intégration des résultats dans un **dashboard interactif** (Streamlit, Dash).  
- Extension à d’autres sources de données (métadonnées, notes utilisateurs).  

---
## 👤 Auteur
📌 Projet développé par **Mickael Gaillard** -->