from sklearn import tree

# Données d'entraînement (diamètre en cm, couleur: 0 = orange, 1 = rouge)
features = [[3.0, 0], [3.2, 0], [3.5, 1], [3.8, 1]]
labels = ["orange", "orange", "pomme", "pomme"]

# Créer et entraîner un arbre de décision
classifier = tree.DecisionTreeClassifier()
classifier.fit(features, labels)

# Test : prédire un fruit rouge de 3.6cm
resultat = classifier.predict([[3.6, 1]])
print("Ce fruit est probablement une :", resultat[0])