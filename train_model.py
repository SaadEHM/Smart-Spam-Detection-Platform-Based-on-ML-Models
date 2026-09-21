"""Entrainement et sauvegarde du meilleur modele de detection de spam."""

import re
import time
import os

import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


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


def main():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'spam.csv'),
                     encoding='latin-1')
    df = df.rename(columns={'v1': 'type', 'v2': 'message'})
    df = df[['type', 'message']]
    df = df.drop_duplicates()
    df = df.dropna(subset=['message'])

    df['label'] = df['type'].map({'ham': 0, 'spam': 1})
    df['message_clean'] = df['message'].apply(clean_text)
    df['message_stemmed'] = df['message_clean'].apply(tokenize_and_stem)

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['message_stemmed'])
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []
    best_accuracy = 0
    best_model_name = ''
    best_model = None

    print('Entrainement et evaluation des modeles :\n')
    print('{:<25} {:>10} {:>10} {:>10} {:>10}'.format(
        'Modele', 'Accuracy', 'Precision', 'Recall', 'F1-Score'))
    print('-' * 70)

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - start

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                        'Recall': rec, 'F1-Score': f1, 'Time': elapsed})

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

        print('{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}'.format(
            name, acc, prec, rec, f1))

    print('\nMeilleur modele :', best_model_name, '(Accuracy: {:.4f})'.format(best_accuracy))

    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf.pkl'))

    meta = {
        'best_model_name': best_model_name,
        'best_accuracy': best_accuracy,
        'results': results,
        'n_samples': len(df),
        'n_spam': int(df['label'].sum()),
        'n_ham': int((df['label'] == 0).sum()),
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, 'meta.pkl'))

    print('\nModele sauvegarde dans :', MODEL_DIR)
    print('Fichiers : best_model.pkl, tfidf.pkl, meta.pkl')


if __name__ == '__main__':
    main()
