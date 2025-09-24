#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy            as np
import pandas           as pd
import matplotlib.pylab as plt
import seaborn          as sns
import plotly.express   as px
from sklearn.feature_extraction.text import TfidfVectorizer  as TFIDF
from sklearn.metrics.pairwise        import linear_kernel


# # **Analyse des données de Netflix**

# ## _Nettoyage_

# In[2]:


data = pd.read_csv('netflix.csv')
data.head()


# In[3]:


data.columns


# In[4]:


# infos détaillées
data.info()


# 'release_year' est en `int`.  
# director, cast, counrty, date_added, rating et duration ont des valeurs manquantes.

# In[5]:


# Que les types de données
data.dtypes


# 1. Enlever les colonnes inutiles.  
# Ici, il n'y en a qu'une.  
# 'Show_idi'.

# In[6]:


data = data.drop('show_id', axis=1)
data


# 2. Retirer  les valeurs manquantes.  
#    6 colonnes sont concernées :
#    * director,
#    * cast,
#    * counrty,
#    * date_added,
#    * rating,
#    * duration.

# In[7]:


# Combiens de valeurs manquentes
valeurs_manquantes = data.isnull().sum()
valeurs_manquantes


# Pour trouver celles qui manquent, il faut faire des recherches basées d'autres valeurs,  
# E.g. le nom du directeur de production, les acteurs ...  
# Ici, les lignes de ces valeurs seront supprimées.  
# <!-- L'influence sur les analyses ne sera pas très grave. -->

# In[8]:


# Suppression des lignes aux valeurs manquantes colonnes par colonnes
data = data.dropna(subset=['country'])
data


# In[9]:


# Manque t-il encore des pays ?
valeurs_manquantes = data.isnull().sum()
valeurs_manquantes


# In[10]:


# Combien de lignes reste t-il ?
data.shape


# In[11]:


# Remplacer les notes manquantes par la plus répendue
# Quel est le mode ?
data['rating'].mode()


# In[12]:


# Sasn l'indice
data['rating'].mode()[0]


# In[13]:


# Remplacer la valeur
data['rating'] = data['rating'].fillna(data['rating'].mode()[0])


# ☝️ Info que je modifie le fichier

# In[14]:


# Manque t-il encore des notes ?
data['rating'].isnull().sum()


# In[15]:


# Nouvelles liste des valeurs mavaleurs manquantes
valeurs_manquantes = data.isnull().sum()
valeurs_manquantes


# In[16]:


# Supprimer la date d'ajout et la durée
data = data.dropna(subset=['date_added', 'duration'])
data.isnull().sum()


# In[17]:


# cast est director ne serviront pas
# Quelle quantité de valeurs unique
data.nunique()


# In[18]:


# Y a-t-il des doublons ?
data.duplicated().sum()


# In[19]:


# Rendre la date exploitableabs
data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')
data.head()


# ☝️ Le message est lié à la gestion des erreurs.

# In[20]:


# Extraire l'année dans une colonne
data['year_added'] = data['date_added'].dt.year.astype('Int64')
data.head()


# In[21]:


# A-t-elle le bon type (int64) ?
data['year_added'].dtypes


# ## _Analyse descriptive_

# La colonne 'rating' doit être expliquée.    
# "`g`" & "`tv-g`" pour tous les âges  
# "`tv-y`" de 2 à 6 ans  
# "`tv-y7`" dés 7 ans  
# "`tv-y7-fv`" recommandé dès 7 ans
# "`pg`" certains passages ne sont pas pour les enfants  
# "`pg-13`" interdit aux moins de 12 ans  
# "`tv-14`" il y a des passages inadaptés au moins de 14 ans  
# "`tv-pg`" inadapté aux jeunes enfants  
# "`R`" les moins de 18 ans sont accompagnés d'un adulte  
# "`tv_ma`" pour les adultes  
# "`nc-17`" public adulte uniquement  
# "`nr`" et "`ur`" non évalué

# In[22]:


# Quel est le pourcentage par catégories ? Utiliser `normalize` pour avoir le pourcentage.
# Avec %matplotlib
plt.figure(figsize=(12,6))
data['rating'].value_counts(normalize=True).plot.bar()
plt.title('Distribution des catégories')
plt.xlabel('Catégories')
plt.ylabel('Fréquence en pourcentage')


# In[23]:


# Via seaborn
sns.catplot(x='rating', data=data, kind='count')
fig = plt.gcf() # get current figure
fig.set_size_inches(12,6)
plt.title('Distribution des catégories')


# In[24]:


# En diagramme circulaire
# autopct ➜ chiffre aprés la virgule / plot.pie ➜ diagramme circulaire
data['rating'].value_counts().plot.pie(autopct= '%1.1f%%', shadow= False, figsize= (15,15))


# ## _Production selon le type_

# In[25]:


custom_palette = ["#1f77b4", "#ff7f0e"]  # Bleu et orange
sns.catplot(x = 'type', data = data, kind = "count", palette=custom_palette)


# In[26]:


custom_palette = ["#1f77b4", "#ff7f0e"]  # Bleu et orange
sns.catplot(
    x='type',
    hue='type',  # Assigner la même variable que `x` à `hue`
    data=data,
    kind="count",
    palette=custom_palette,
    legend=False  # Désactiver la légende si elle n'est pas nécessaire
)     # Contemporain, très long


# In[27]:


# La fréquence en texte
data['type'].value_counts(normalize=True)


# In[28]:


# En pourcentage au lieu d'entre zéro et un
data['type'].value_counts(normalize=True)*100


# In[29]:


# Afficher en diagramme circulaire (explode sépare les éléments)
data['type'].value_counts().plot.pie(autopct = '%1.1f%%', shadow = False, cmap = 'summer', figsize = (5,5), explode = [0,0.1])
plt.title('Les types de programmes', fontsize = 16)
plt.legend()


# ## _Productions ajoutées par année_

# In[30]:


plt.figure(figsize=(5,6))
data['year_added'].value_counts().plot.bar(color='darkgreen')
plt.title('Ajout de production')
plt.xlabel('Années')
plt.ylabel('Productions')


# ## _Tendance par an (tv ou films)_

# In[31]:


# En pourcentage
data.groupby('year_added')['type'].value_counts(normalize=True)*100


# In[32]:


plt.figure(figsize=(12,5))
sns.countplot(x='year_added', hue='type', data=data)
plt.title('Procution par tyupe et année')


# In[33]:


# Calcul des effectifs par année et par type
count_df = (
    data.groupby(['year_added', 'type']).size()
    .reset_index(name='count')
)

# Calcul du total par année
total_per_year = count_df.groupby('year_added')['count'].transform('sum')

# Calcul du pourcentage
count_df['percentage'] = count_df['count'] / total_per_year * 100

# Tableau croisé pour le plot
pivot = count_df.pivot(index='year_added', columns='type', values='percentage').fillna(0)
pivot = pivot.sort_index()

# Couleurs vert et orange hexadécimal
colors = ['green', '#fc4b08']

ax = pivot.plot(kind='bar', stacked=True, figsize=(12,7), color=colors)

# Ajouter les pourcentages au centre de chaque segment en vertical
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    if height > 0:  # ne pas afficher si segment vide
        ax.text(
            x + width / 2,
            y + height / 2,
            f'{height:.1f}%',
            ha='center',
            va='center',
            rotation=90,       # Rotation verticale (90 degrés)
            color='black',
            fontsize=14,
            fontweight='bold'
        )

plt.title('Répartition en pourcentage des types de productions par année')
plt.xlabel('Année ajoutée')
plt.ylabel('Pourcentage (%)')
plt.legend(title='Type de production', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## _Tendance des acquisitions de 2011 à 2021_

# La bibliothèque `Plotly` est un outil de visualisation, générant des données géographiques, scientifiques, statistiques et financières.  
# Il fait les tracés avec moins de lignes de codes de `Matplotlib`.

# In[34]:


acquisitions = data.groupby(['year_added','type']).size().reset_index(name = 'Productions')
acquisitions


# Depuis 2011, le nombre devient intéressant, l'analyse commencera là.

# In[35]:


acquisitions = acquisitions[acquisitions['year_added']>=2011]
acquisitions


# In[37]:


acquisition_graph = px.line(acquisitions, x ="year_added", y="Productions", color = 'type', title = "Tendance d'ajout/année")
acquisition_graph.show()


# # **Productions de chaque type selon la catégorie d'évaluation**

# In[36]:


plt.figure(figsize=(12,5))
sns.countplot(x='rating', hue='type', data=data)
plt.title("Productions de chaque type selon la catégorie d'évaluation")


# Pour les enfants, il y a plus de série, dans les autres catégories, les films sont largement dominants,

# # **Productions par pays**

# In[38]:


# `expend` pour séparer les éléments en colonne
pays = data.set_index('title').country.str.split(',', expand=True)
pays


# In[39]:


pays = data.set_index('title').country.str.split(',', expand=True).stack() # pivote les colonnes de l'index
pays.head(25)


# `reset_index(level=1, drop=True)` : seul le second niveau est réutilisé ('level=1') de l’index (le numéro de colonne créé par le stack), puis le défalque grâce à `drop=True`. L’index final contient uniquement les titres ('title'), chaque pays de la liste étant maintenant une ligne distincte.

# In[40]:


pays = data.set_index('title').country.str.split(',', expand=True).stack().reset_index(level=1,drop=True)
pays.head(25)


# In[41]:


# les 25 pays qui produisent le plus
top_25_pays = pays.value_counts().head(25)
top_25_pays


# In[42]:


# Création d'un graphique en barres horizontales avec Seaborn
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=top_25_pays.values, y=top_25_pays.index, palette='viridis')
plt.title('Top 25 des pays producteurs')
plt.xlabel("Nombre d'occurrences")
plt.ylabel("Pays")

# Ajouter les valeurs sur chaque barre
for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3)

plt.tight_layout()
plt.show()


# In[43]:


# Le top 15

top15 = pays.reset_index(name='pays')
plt.figure(figsize=(16, 8))
ax = sns.countplot(x='pays', data=top15, order=pays.value_counts().index[:15], palette='turbo')

# Fond noir
plt.gcf().patch.set_facecolor('black')  # Fond de la figure
ax.set_facecolor('black')               # Fond du graphe
plt.title("Les 15 pays les plus producteurs", fontsize=16, color='white')
plt.xlabel('Pays', color='white')
plt.ylabel('Productions', color='white')

# Supprimer les noms de pays sous les barres
"""ax.set_xticklabels([])"""

# Afficher le nom du pays dans la barre, centré et à la verticale
"""for p, label in zip(ax.patches, pays.value_counts().index[:15]):
    height = p.get_height()
    x = p.get_x() + p.get_width() / 2
    y = height / 2
    ax.text(x, y, label, ha='center', va='center', rotation=90, color='white', fontsize=14)"""

# Afficher aussi le nombre de production proche du haut de la barre
for p in ax.patches:
    height = p.get_height()
    x = p.get_x() + p.get_width() / 2
    y = height * 0.9
    ax.text(x, y, int(height), ha='center', va='center', rotation=45, color='white', fontsize=14)
    ax.tick_params(colors='white') # couleur des pays
plt.show()


# # **Productions par pays selon le type**

# In[44]:


data_producteurs = data[(data['country']=="United States")|(data['country']=="India")
                      |(data['country']=="United Kingdom")|(data['country']=="Canada")
                      |(data['country']=="France")|(data['country']=="Japan")
                      |(data['country']=="Spain")|(data['country']=="South Korea")
                      |(data['country']=="Germany")|(data['country']=="Mexico")]
data_producteurs


# 5269 productions sont issues des 10 pays les plus producteurs.

# In[45]:


# Combien de film et émission ?
plt.figure(figsize = (11,6))
sns.countplot(x = 'country', hue = 'type', data = data_producteurs)


# Le Royaume-Uni, le Japon et la Corée du Sud produisent plus d'émissions.

# # **Top 10 des catégories de productions**

# In[46]:


top_categorie= data.set_index('title').listed_in.str.split(', ', expand = True)
top_categorie


# Il y a trois catégories maximum par production.
# 
# Indexation à plusieurs niveaux.

# In[47]:


top_categorie = data.set_index('title').listed_in.str.split(', ', expand = True).stack()
top_categorie


# In[48]:


# Initialiser l'index de la série
top_categorie = data.set_index('title').listed_in.str.split(', ', expand = True).stack().reset_index(level = 1, drop = True)
top_categorie


# In[49]:


# Représenter les catégories
plt.figure(figsize = (12,14))
sns.countplot(y = top_categorie)
plt.title('Classement des catégories', fontsize = 16)


# In[50]:


top10 = top_categorie.value_counts().index[:10]
top10


# In[51]:


# Extraction des 10 catégories les plus fréquentes

# Réinitialiser l'index (pour éviter les doublons dans l'index)
top_categorie = top_categorie.reset_index(drop=True)

plt.figure(figsize=(12, 8))

# Pas de 'hue' !
sns.countplot(y=top_categorie, order=top10, palette='turbo') # hue=top_categorie

plt.title('Top 10 des catégories', fontsize=16)
plt.xlabel('Nombre d\'occurrences')
plt.ylabel('Catégorie')
plt.show()


# # **Machine learning**

# In[53]:


# Nettoyage des genres (pas de valeurs manquantes, mais on s'assure que c'est une chaîne de caractères)
data = data.copy()  # Crée une copie explicite
data['genres'] = data['listed_in'].astype(str)


# In[55]:


# Vectorisation TF-IDF
tfidf = TFIDF(stop_words='english')                  # ignorer les mots courants en anglais
tfidf_matrix = tfidf.fit_transform(data['genres'])   # cf. ci-dessous


# fit_transform fait deux choses :
# * fit : Apprend le vocabulaire et les poids TF-IDF à partir des données (data['genres']).
# * transform : Convertit les textes en une matrice de caractéristiques TF-IDF.

# In[62]:


# Calcul de la similarité cosinus
indices = pd.Series(data.index, index=data['title']).drop_duplicates()
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[64]:


# Créer une série pour mapper les titres aux indices
indices = pd.Series(data.index, index=data['title']).drop_duplicates()


# ### _Les recommandations_

# In[65]:


# Tableau de recommandations

# Fonction de recommandation
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Top 5 recommandations
        movie_indices = [i[0] for i in sim_scores]
        return data[['title', 'listed_in', 'country']].iloc[movie_indices]
    except KeyError:
        return f"Le titre '{title}' n'est pas dans le dataset."

# Fonction pour afficher les recommandations
def plot_recommendations(title):
    try:
        recommendations = get_recommendations(title)
        print("Recommandations :", recommendations)  # Debug

        if isinstance(recommendations, str):
            print(recommendations)
        else:
            table_data = recommendations[['title', 'listed_in', 'country']]
            print("Données du tableau :", table_data)  # Debug

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis('off')
            ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')
            plt.title(f"Top 5 Recommandations pour : {title}")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Erreur : {e}")

# Exemple d'utilisation
plot_recommendations("Dick Johnson Is Dead")


# In[66]:


# Test
get_recommendations("The Dark Knight")


# In[68]:


get_recommendations("Final Account")

