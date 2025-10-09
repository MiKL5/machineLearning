#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas                  as     pd
from   sklearn.model_selection import train_test_split
from   sklearn.svm             import SVR
from   sklearn.metrics         import mean_absolute_error
from   sklearn.linear_model    import LinearRegression
from   sklearn.pipeline        import Pipeline
from   sklearn.preprocessing   import StandardScaler


# In[ ]:


import pandas as pd
df = pd.read_csv('Student_Performance.csv')
df


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


y = df['Performance Index'].copy()
X = df.drop(columns='Performance Index').copy()
X.head()


# In[9]:


X = pd.get_dummies(X, columns=['Extracurricular Activities'])
X.head()


# ## **Division en features et labels**

# In[10]:


y = df['Performance Index'].copy()
X = df.drop(columns='Performance Index').copy()
X.head()


# La colonne catégorielle 'Extracurricular Activities' est dans les features.
# La technique 'encodage One-Hot' pour gérer ça.

# In[11]:


X = pd.get_dummies(X, columns=['Extracurricular Activities'])
X.head()


# ## **Fractionner pour l'entraînement  et le test**

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# ## **Modélisation**

# ### Support Vector Regressor (SVR)

# In[13]:


svr = SVR()
svr.fit(X_train, y_train)
svr_prediction = svr.predict(X_test)

svr_mae = mean_absolute_error(y_test, svr_prediction)
svr_mae


# Trés faible erreur ; c'est plutôt bien.

# ### Régression linéaire

# In[14]:


lr = LinearRegression()
lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_prediction)
lr_mae


# Bien.

# ### Normaliser les données

# In[15]:


lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
lr_pipeline_prediction = lr_pipeline.predict(X_test)
lr_pipeline_mae = mean_absolute_error(y_test, lr_pipeline_prediction)
lr_pipeline_mae


# La normalisation a très lègerement améliorer.
