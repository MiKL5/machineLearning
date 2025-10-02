#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd
import numpy   as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity


# In[2]:


movies = pd.read_csv('movies.csv')


# In[3]:


movies.head()


# In[4]:


movies.shape


# In[5]:


movies.isnull().sum()


# In[6]:


needed_cols = ['genres', 'keywords', 'tagline', 'cast', 'director']
df = movies[needed_cols]

df.isnull().sum()


# In[7]:


for col in needed_cols:
  df[col] = df[col].fillna('')

df.isnull().sum()


# In[8]:


# Combiner les features
combined_cols = df['genres'] + " " + df['keywords'] + " " + df['tagline'] + " " + df['cast'] + " " + df['director']
combined_cols


# In[9]:


# Que contient la première ligne ?
combined_cols[0]


# In[10]:


# convertir en nombres en utilisant la fonction `TfidfVectorizer()`
vectorizer = TfidfVectorizer()
vectorized_combined_cols = vectorizer.fit_transform(combined_cols)


# In[11]:


# À quoi ça ressemble ?
np.set_printoptions(threshold=np.inf)
vectorized_combined_cols.toarray()[0]


# In[12]:


similarity = cosine_similarity(vectorized_combined_cols)
np.set_printoptions(threshold=10)   # n'imprimera que les 10 premières lignes du tableau numpy
similarity


# In[13]:


# Lister les films
movie_titles = movies['title'].tolist()
movie_titles


# In[14]:


# Trouver les mots analogues
user_input = input("Entre ton film préféré pour obtenir des recommandations similaires : ")
close_movie_names = difflib.get_close_matches(user_input, movie_titles)
close_movie_names


# In[15]:


# Les premiers noms sont plus pertinents
close_match = close_movie_names[0]
close_match


# In[16]:


# Quelle est la valeur de l'index du film ?
index_of_the_movie = movies[movies.title == close_match]['index'].values[0]
index_of_the_movie


# In[17]:


# Quel est le score de similarité ?
similarity_score = list(enumerate(similarity[index_of_the_movie]))
similarity_score


# In[18]:


# Trier par décroissancce pour que les similarités soient en haut de la liste
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
sorted_similar_movies


# In[19]:


# affichier les 30 films analogues à Iron man 2
print("Films pouvants t'intéresser : \n")
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies[movies.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1

