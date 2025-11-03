#%%
import pandas                    as     pd
from   mlxtend.preprocessing     import TransactionEncoder
from   mlxtend.frequent_patterns import fpgrowth
#%% md
# # **Associer les règles d'apprentissage avec FP-Growth**
#%%
dataset = [['Lait', 'Farine', 'Biscuit', 'Pain', 'Oeufs', 'Banane'],
           ['Café', 'Farine', 'Biscuit', 'Pain', 'Oeufs', 'Banane'],
           ['Lait', 'Sel', 'Pain', 'Oeufs'],
           ['Lait', 'Licorne', 'Sel', 'Pain', 'Banane'],
           ['Sel', 'Farine', 'Sucre', 'Pain', 'Glace', 'Oeufs']]
#%% md
# ## **Faire un prétraitement**
# 
# L'outil '`TransactionEncoder`' de la bibliothèque '`lmxtend`', fonctionne comme '`OneHotEncoder`'.
#%%
te = TransactionEncoder()
te_array = te.fit_transform(dataset)
te_array
#%% md
# C'est un tableau de valeurs booléennes
#%%
te.columns_
#%%
te.columns_mapping_
#%% md
# ## **Faire du tableau un dataframe**
#%%
df = pd.DataFrame(te_array, columns=te.columns_)
df
#%% md
# ## **Appliquer l'algorithme FP-Growth**
# 
# Le seuil est à 60 %.
#%%
fpgrowth(df, min_support=0.6)
#%% md
# ## **Remplacer 'Itemset' par la valeur originelle**
#%%
fpgrowth(df, min_support=0.6, use_colnames=True)
#%% md
# Tous les clients achète du pain.
# 80 % des clients achètes ensembles les oeufs et le pain.
# Les articles (lait, pain, oeuf, farine, sel et banane) doivent êtres à proximité.