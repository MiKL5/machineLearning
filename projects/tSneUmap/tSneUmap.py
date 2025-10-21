#%%
import plotly.express  as px
import seaborn         as sns
from  sklearn.manifold import TSNE
#%% md
# # **Réduire de la dimensionnalité par t-SNE et de UMAP**
#%%
df = px.data.iris()
df.head()
#%% md
# ## **Dans la variable 'seatures', ne pas stocker 'species_id'**
#%%
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
fig = px.scatter_matrix(df, dimensions=features, color='species')
fig.show()
#%% md
# ## **Faire une graphique aved Seaborn**
#%%
df2 = df.drop(columns='species_id')
sns.pairplot(df2, hue='species')
#%% md
# La bibliothèque Plotly, peréet de voir les valeurs et de zoomer;
#%% md
# ## **Créer un modèle T-SNE**
#%%
features = df.loc[:, :'petal_width']
tsne = TSNE(n_components=2, random_state=7)
projections = tsne.fit_transform(features)
fig = px.scatter(projections, x=0, y=1, color=df.species, labels={'color':'species'})
fig.show()
#%% md
# Transformer tous les ensembles de données que nous lui transmettons en ensembles bidimensionnels ; le paramètre '`n_components`' est fixé à 2.
# les features sont transmises au modèle TSNE et stocké les features 2D transformées dans une variable 'projections'.
# Plotly, a créé un diagramme de dispersion des features 2D (projections).
#%% md
# ## **Créer une feature 3D par '`n_components`'**
#%%
tsne = TSNE(n_components=3, random_state=7)
projections = tsne.fit_transform(features)
fig = px.scatter_3d(projections, x=0, y=1, z=2, color=df.species, labels={'color':'species'})
fig.show()