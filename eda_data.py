"""Calculs EDA, generation d'images et preparation des donnees ML."""

import os
import re
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_FILE = os.path.join(BASE_DIR, '..', 'spam.csv')

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Registre unique des modeles (cle UI -> nom affiche -> classe)
ALGO_CLASSES = {
    'naive_bayes': MultinomialNB(),
    'logistic': LogisticRegression(max_iter=1000, random_state=42),
    'svm': SVC(kernel='linear', probability=True, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'decision_tree': DecisionTreeClassifier(random_state=42),
}

ALGO_NAMES = {
    'naive_bayes': 'Naive Bayes',
    'logistic': 'Logistic Regression',
    'svm': 'SVM',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'knn': 'KNN',
    'decision_tree': 'Decision Tree',
}

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

_cache = {}


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'#', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_stem(text):
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def load_clean_data():
    if 'df' in _cache:
        return _cache['df']
    df = pd.read_csv(DATA_FILE, encoding='latin-1')
    df = df.rename(columns={'v1': 'type', 'v2': 'message'})
    df = df[['type', 'message']]
    df = df.drop_duplicates()
    df = df.dropna(subset=['message'])
    df['label'] = df['type'].map({'ham': 0, 'spam': 1})
    df['message_clean'] = df['message'].apply(clean_text)
    df['message_stemmed'] = df['message_clean'].apply(tokenize_and_stem)
    _cache['df'] = df
    return df


def get_data_split():
    if 'split' in _cache:
        return _cache['split']
    df = load_clean_data()
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['message_stemmed'])
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    _cache['split'] = (tfidf, X, y, X_train, X_test, y_train, y_test, X_train_sm, y_train_sm)
    return _cache['split']


def get_tfidf():
    """Retourne le vectoriseur TF-IDF entraine (mis en cache)."""
    return get_data_split()[0]


def get_smote_stats():
    """Retourne les stats avant/apres SMOTE pour l'EDA."""
    _, _, _, _, _, y_train, _, _, y_train_sm = get_data_split()
    before = {'ham': int((y_train == 0).sum()), 'spam': int((y_train == 1).sum())}
    after = {'ham': int((y_train_sm == 0).sum()), 'spam': int((y_train_sm == 1).sum())}
    return before, after


def get_eda_json():
    df = load_clean_data()

    class_counts = df['type'].value_counts()
    n_spam = int(class_counts.get('spam', 0))
    n_ham = int(class_counts.get('ham', 0))

    avg_words_spam = df[df['type'] == 'spam']['message'].apply(lambda x: len(str(x).split())).mean()
    avg_words_ham = df[df['type'] == 'ham']['message'].apply(lambda x: len(str(x).split())).mean()

    spam_clean = ' '.join(df[df['type'] == 'spam']['message_stemmed']).split()
    spam_clean = [w for w in spam_clean if len(w) > 2]
    top_spam = Counter(spam_clean).most_common(20)

    ham_clean = ' '.join(df[df['type'] == 'ham']['message_stemmed']).split()
    ham_clean = [w for w in ham_clean if len(w) > 2]
    top_ham = Counter(ham_clean).most_common(20)

    return {
        'n_samples': int(len(df)),
        'n_columns': int(df.shape[1]),
        'n_missing': int(df.isna().sum().sum()),
        'n_spam': n_spam,
        'n_ham': n_ham,
        'spam_pct': round(n_spam / len(df) * 100, 1),
        'ham_pct': round(n_ham / len(df) * 100, 1),
        'avg_words_spam': round(avg_words_spam, 1),
        'avg_words_ham': round(avg_words_ham, 1),
        'top_spam_words': [w for w, _ in top_spam],
        'top_spam_counts': [c for _, c in top_spam],
        'top_ham_words': [w for w, _ in top_ham],
        'top_ham_counts': [c for _, c in top_ham],
    }


def generate_wordclouds():
    df = load_clean_data()

    spam_text = ' '.join(df[df['type'] == 'spam']['message_clean'])
    wc_spam = WordCloud(width=1200, height=500, background_color='white',
                        colormap='Reds', max_words=100, random_state=42)
    wc_spam.generate(spam_text)
    wc_spam.to_file(os.path.join(STATIC_DIR, 'wc_spam.png'))

    ham_text = ' '.join(df[df['type'] == 'ham']['message_clean'])
    wc_ham = WordCloud(width=1200, height=500, background_color='white',
                       colormap='Blues', max_words=100, random_state=42)
    wc_ham.generate(ham_text)
    wc_ham.to_file(os.path.join(STATIC_DIR, 'wc_ham.png'))


def generate_length_histogram():
    df = load_clean_data()
    df['_len'] = df['message'].apply(len)

    fig, ax = plt.subplots(figsize=(10, 5))
    df[df['type'] == 'ham']['_len'].plot(kind='hist', bins=40, alpha=0.6,
                                         color='royalblue', label='Ham', ax=ax, density=True)
    df[df['type'] == 'spam']['_len'].plot(kind='hist', bins=40, alpha=0.6,
                                          color='crimson', label='Spam', ax=ax, density=True)
    ax.set_xlabel('Longueur du message (caracteres)')
    ax.set_ylabel('Densite')
    ax.set_title('Distribution de la longueur des messages', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, 'len_hist.png'), dpi=150)
    plt.close(fig)


def generate_all_images():
    generate_wordclouds()
    generate_length_histogram()
    generate_smote_chart()
    generate_ml_charts()


def generate_smote_chart():
    """Graphique avant/apres SMOTE (balancing des classes)."""
    before, after = get_smote_stats()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cats = ['ham', 'spam']
    colors = ['royalblue', 'crimson']

    axes[0].bar(cats, [before['ham'], before['spam']], color=colors)
    axes[0].set_title('Avant SMOTE (entrainement)', fontweight='bold')
    for i, v in enumerate([before['ham'], before['spam']]):
        axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

    axes[1].bar(cats, [after['ham'], after['spam']], color=colors)
    axes[1].set_title('Apres SMOTE (entrainement)', fontweight='bold')
    for i, v in enumerate([after['ham'], after['spam']]):
        axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')

    for ax in axes:
        ax.set_ylabel('Nombre de messages')

    fig.suptitle('Balancing des classes (SMOTE)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, 'smote_chart.png'), dpi=150)
    plt.close(fig)


def train_all_models():
    """Entraine les 7 modeles sur les donnees SMOTE (comme le notebook) et les met en cache."""
    if 'ml' in _cache:
        return _cache['ml']

    tfidf, X, y, X_train, X_test, y_train, y_test, X_train_sm, y_train_sm = get_data_split()

    results = []
    fitted = {}
    best_acc = 0
    best_name = ''
    best_model = None

    for key, model in ALGO_CLASSES.items():
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        results.append({
            'Model': ALGO_NAMES[key],
            'Accuracy': round(acc, 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
        })
        fitted[ALGO_NAMES[key]] = model
        if acc > best_acc:
            best_acc = acc
            best_name = ALGO_NAMES[key]
            best_model = model

    df_results = (pd.DataFrame(results)
                  .sort_values('Accuracy', ascending=False)
                  .reset_index(drop=True))
    _cache['ml'] = (df_results, fitted, best_name, best_model, X_test, y_test)
    return _cache['ml']


def get_results_table():
    """Tableau comparatif des resultats de tous les modeles."""
    df_results, *_ = train_all_models()
    return df_results.to_dict(orient='records')


def generate_ml_charts():
    """Genere tous les graphes de la partie ML du notebook."""
    df_results, fitted, best_name, best_model, X_test, y_test = train_all_models()

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    # 1. Barres groupees des performances (cellule 6.5)
    x = np.arange(len(df_results))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df_results[metric], width, label=metric,
               color=colors[i], alpha=0.85)
    ax.set_xlabel('Modele')
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des performances des modeles', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_results['Model'], rotation=30, ha='right')
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, 'perf_bar.png'), dpi=150)
    plt.close(fig)

    # 2. Heatmap des performances
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_results.set_index('Model')[metrics], annot=True, fmt='.3f',
                cmap='RdYlGn', linewidths=0.5, ax=ax, vmin=0.7, vmax=1.0)
    ax.set_title('Heatmap des performances', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, 'perf_heatmap.png'), dpi=150)
    plt.close(fig)

    # 3. Courbes ROC de tous les modeles
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, model in fitted.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label='{} (AUC={:.3f})'.format(name, auc), linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Courbes ROC - Tous les modeles', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, 'roc_all.png'), dpi=150)
    plt.close(fig)

    # 4. Matrice de confusion du meilleur modele
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[0].set_xlabel('Prediction')
    axes[0].set_ylabel('Reel')
    axes[0].set_title('Matrice de confusion - ' + best_name, fontweight='bold')

    errors = int((y_pred_best != y_test).sum())
    correct = int((y_pred_best == y_test).sum())
    axes[1].bar(['Correct', 'Erreur'], [correct, errors],
                color=['#4CAF50', '#F44336'], alpha=0.8)
    axes[1].set_ylabel('Nombre')
    axes[1].set_title('Resultats du meilleur modele', fontweight='bold')
    for i, v in enumerate([correct, errors]):
        axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, 'confusion_best.png'), dpi=150)
    plt.close(fig)


def train_algorithm(algo_name):
    """Entraine l'algorithme choisi sur les donnees SMOTE et le sauvegarde."""
    if algo_name not in ALGO_CLASSES:
        raise ValueError('Algorithme inconnu : ' + algo_name)

    tfidf, X, y, X_train, X_test, y_train, y_test, X_train_sm, y_train_sm = get_data_split()

    model = ALGO_CLASSES[algo_name]
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'f1': round(f1_score(y_test, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
    }

    joblib.dump(model, os.path.join(MODEL_DIR, 'selected_model.pkl'))
    meta = joblib.load(os.path.join(MODEL_DIR, 'meta.pkl'))
    meta['current_model'] = ALGO_NAMES[algo_name]
    meta['current_accuracy'] = metrics['accuracy']
    meta['current_metrics'] = metrics
    joblib.dump(meta, os.path.join(MODEL_DIR, 'meta.pkl'))

    return model, metrics, cm.tolist()


def generate_curves():
    """Genere la courbe ROC et Precision-Recall avec le modele selectionne."""
    model_path = os.path.join(MODEL_DIR, 'selected_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    model = joblib.load(model_path)

    tfidf, X, y, X_train, X_test, y_train, y_test, X_train_sm, y_train_sm = get_data_split()
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color='#2196F3',
            label='ROC (AUC = {:.3f})'.format(auc))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Courbe ROC', fontweight='bold')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, 'roc_curve.png'), dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, color='#4CAF50')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Courbe Precision-Recall', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, 'pr_curve.png'), dpi=150)
    plt.close(fig)
