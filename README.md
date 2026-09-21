# Plateforme Intelligente de Detection de Spam

Application web Flask multi-pages pour la **detection de mails spam** avec **EDA** (Exploratory Data Analysis) et **Machine Learning**.

## Structure

```
plateforme_spam/
├── app.py                # Application Flask (routes pages + API)
├── database.py           # SQLite pour l'historique des predictions
├── eda_data.py           # Stats EDA, generation d'images, entrainement
├── train_model.py        # Entrainement initial du meilleur modele (SVM)
├── spam_history.db       # Base SQLite (creee au premier lancement)
├── models/
│   ├── best_model.pkl    # Meilleur modele (SVM, accuracy 97.97%)
│   ├── tfidf.pkl         # Vectoriseur TF-IDF
│   ├── meta.pkl          # Metadonnees du modele
│   └── selected_model.pkl# Modele entraine via l'interface (optionnel)
├── templates/
│   ├── base.html         # Layout (navbar + sidebar)
│   ├── accueil.html      # Page d'accueil avec statistiques
│   ├── eda.html          # Analyse exploratoire des donnees
│   ├── preprocessing.html# Pipeline de nettoyage du texte
│   ├── entrainement.html # Choix + entrainement d'un algorithme
│   ├── test.html         # Tester un email (spam / ham)
│   ├── statistiques.html # ROC, Precision-Recall
│   └── historique.html   # Historique des predictions (SQLite)
└── static/
    ├── style.css         # Styles
    ├── script.js         # Helpers Chart.js
    ├── wc_spam.png       # WordCloud spam
    ├── wc_ham.png        # WordCloud ham
    ├── len_hist.png      # Histogramme longueur
    ├── corr_heatmap.png  # Heatmap correlations
    ├── roc_curve.png     # Courbe ROC
    └── pr_curve.png      # Courbe Precision-Recall
```

## Installation

```bash
pip install flask scikit-learn nltk joblib pandas matplotlib seaborn wordcloud imbalanced-learn
```

## Entrainement initial

```bash
python train_model.py
```

Entraine 7 algorithmes (Logistic Regression, Naive Bayes, SVM, Random Forest, KNN, Decision Tree, Gradient Boosting) et sauvegarde le meilleur (SVM, 97.97%).

## Demarrage

```bash
python app.py
```

Ouvrir http://127.0.0.1:5000

## Pages

| Page | Description |
|------|-------------|
| **Accueil** | Presentation du projet + stats (nb emails, spam, ham, accuracy) |
| **EDA** | Stats dataset, repartition spam/ham, moyennes, histogrammes, wordclouds, top mots, heatmap |
| **Preprocessing** | Pipeline complet avec le texte apres chaque etape |
| **Entrainement** | Choisir un algorithme, entrainer, afficher metrics + matrice de confusion |
| **Test** | Coller un email -> prediction spam/ham avec probabilite + etapes de traitement |
| **Statistiques** | Courbes ROC et Precision-Recall |
| **Historique** | Predictions enregistrees dans SQLite |
