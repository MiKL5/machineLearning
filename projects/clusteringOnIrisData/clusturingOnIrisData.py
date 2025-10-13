#%%
from   sklearn.cluster         import KMeans
import pandas                  as     pd
from   sklearn.datasets        import load_iris
import matplotlib.pyplot       as     plt
import seaborn                 as     sns
from   sklearn.metrics         import silhouette_score
import plotly.express          as     px
from   sklearn.metrics         import adjusted_rand_score
from   sklearn.preprocessing   import StandardScaler
from   sklearn.decomposition   import PCA
from   sklearn.cluster         import DBSCAN
from   sklearn.cluster         import AgglomerativeClustering
from   scipy.cluster.hierarchy import linkage, dendrogram
import joblib
#%%
iris = load_iris()
#%%
X = iris.data
#%%
# créer un modèle de clustering KMean
kmeans = KMeans(n_clusters=3, random_state=42)
#%%
# La feature dans le modèle
kmeans.fit(X)
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

kmeans.labels_
#%%
# Les étiquettes dans le dataframe
iris_df['cluster'] = kmeans.labels_
#%%
sns.scatterplot(x=iris_df['sepal length (cm)'], y=iris_df['sepal width (cm)'], hue=iris_df['cluster'], palette='viridis')
plt.xlabel('Longueur du sépale (cm)')
plt.ylabel('Largeur du sépale (cm)')
plt.title("K-Mean Clustering de l'ensemble de données Iris")
#%%
# Évaluation de la Qualité du Clustering
print(f"Inertie du modèle: {kmeans.inertia_}")
#%%
# Mesure la cohésion et la séparation des clusters (valeur entre -1 et 1, plus proche de 1 = meilleur).
score = silhouette_score(X, kmeans.labels_)
print(f"La silhouette Score est {score:.2f}")
#%%
# Quel est le nombre optimal de clusters (Elbow Method (Méthode du Coude)) ?
inertias = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 8), inertias, marker='o')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.title("Méthode du Coude pour K-Means")
#%%
# Visualiser en 3D pour mieux voir la séparation des clusters
fig = px.scatter_3d(
    iris_df,
    x='sepal length (cm)',
    y='sepal width (cm)',
    z='petal length (cm)',
    color='cluster',
    title="Clustering K-Means 3D sur Iris"
)
fig.show()
#%%
# Visualiser les relations entre toutes les paires de features
sns.pairplot(iris_df, hue='cluster', palette='viridis')
plt.suptitle("Pair Plot des Features par Cluster", y=1.02)
#%%
# Afficher au graphique les centroïdes des clusters | Quelle est la position des points centraux ? (comprendre la structure des données et d'interpréter les résultats du clustering)
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=iris.feature_names)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='cluster', palette='viridis')
sns.scatterplot(data=centroids_df, x='sepal length (cm)', y='sepal width (cm)', color='red', marker='X', s=200, label='Centroïdes')
plt.legend()
plt.title("K-Means avec Centroïdes")
#%%
# Comparer avec les vrais labels
ari = adjusted_rand_score(iris.target, kmeans.labels_)
print(f"Adjusted Rand Index: {ari:.2f}")  # 1 = parfait, 0 = aléatoire
#%%
# Visualiser Side-by-Side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='cluster', palette='viridis', ax=axes[0])
axes[0].set_title("Clusters K-Means")

# Visualiser avec les vrais labels
iris_df['species'] = iris.target
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='species', palette='viridis', ax=axes[1])
axes[1].set_title("Vrais Labels (Espèces)")
plt.show()
#%%
# Normaliser les donées
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_scaled = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
iris_df['cluster_scaled'] = kmeans_scaled.labels_
#%%
# Utiliser PCA pour une visualisation en 2D plus claire
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

iris_df['pca1'] = X_pca[:, 0]
iris_df['pca2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='cluster_scaled', palette='viridis')
plt.title("Clusters après PCA")
#%%
# Explorer avec l'algo DBSCQN
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
iris_df['dbscan_cluster'] = dbscan.labels_

plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='dbscan_cluster', palette='viridis')
plt.title("Clustering avec DBSCAN")
#%%
agg = AgglomerativeClustering(n_clusters=3).fit(X_scaled)
iris_df['agg_cluster'] = agg.labels_

Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogramme du Clustering Hiérarchique")
#%%
# Interpréter la biologie
iris_df.groupby('cluster').mean()
#%%
# Déployer et partager
joblib.dump(kmeans, 'kmeans_iris.pkl')