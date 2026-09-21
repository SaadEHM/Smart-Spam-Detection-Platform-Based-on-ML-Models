"""Plateforme intelligente de detection de spam - Application Flask multi-pages."""

import os
import re

import joblib
from flask import Flask, render_template, request, jsonify

import eda_data
import database

app = Flask(__name__)
database.init_db()

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Etat des modeles en memoire (recharge apres chaque entrainement)
_model = None
_meta = None


def load_models():
    global _model, _meta
    _model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    _meta = joblib.load(os.path.join(MODEL_DIR, 'meta.pkl'))


def reload_models():
    """Recharge le modele et les metadonnees depuis le disque (apres entrainement)."""
    global _model, _meta
    selected = os.path.join(MODEL_DIR, 'selected_model.pkl')
    if os.path.exists(selected):
        _model = joblib.load(selected)
    else:
        _model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    _meta = joblib.load(os.path.join(MODEL_DIR, 'meta.pkl'))


load_models()


def get_model():
    if _model is None:
        reload_models()
    return _model


def get_meta():
    return _meta


def pipeline_steps(message):
    """Retourne le texte apres chaque etape du preprocessing."""
    steps = []
    steps.append(('Message brut', message))
    text = message.lower()
    steps.append(('Mise en minuscules', text))
    text = re.sub(r'#', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    steps.append(('Suppression URLs', text))
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    steps.append(('Suppression HTML / emails', text))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    steps.append(('Suppression chiffres et caracteres speciaux', text))
    tokens = text.split()
    tokens = [t for t in tokens if t not in eda_data.stop_words and len(t) > 2]
    steps.append(('Suppression stopwords', ' '.join(tokens)))
    tokens = [eda_data.stemmer.stem(t) for t in tokens]
    steps.append(('Stemmisation', ' '.join(tokens)))
    return steps


def predict(message):
    cleaned = eda_data.clean_text(message)
    stemmed = eda_data.tokenize_and_stem(cleaned)
    vectorized = eda_data.get_tfidf().transform([stemmed])
    model = get_model()
    proba = model.predict_proba(vectorized)[0]
    label = 'spam' if proba[1] >= 0.5 else 'ham'
    return {
        'label': label,
        'confidence': round(max(proba) * 100, 2),
        'prob_spam': round(proba[1] * 100, 2),
        'prob_ham': round(proba[0] * 100, 2),
    }


@app.route('/')
def accueil():
    eda = eda_data.get_eda_json()
    total_h, n_spam_h = database.count_predictions()
    return render_template('accueil.html',
                           meta=get_meta(), eda=eda, total_history=total_h, history_spam=n_spam_h)


@app.route('/eda')
def eda_page():
    eda_data.generate_all_images()
    eda = eda_data.get_eda_json()
    smote_before, smote_after = eda_data.get_smote_stats()
    return render_template('eda.html', eda=eda, smote_before=smote_before, smote_after=smote_after)


@app.route('/preprocessing')
def preprocessing_page():
    example = "WINNER!! As a valued network customer you have been selected to receive a prize! Call 0871 170 6024 now."
    steps = pipeline_steps(example)
    return render_template('preprocessing.html', steps=steps, example=example)


@app.route('/train')
def train_page():
    return render_template('entrainement.html', meta=get_meta())


@app.route('/test')
def test_page():
    return render_template('test.html')


@app.route('/stats')
def stats_page():
    eda_data.generate_curves()
    results = eda_data.get_results_table()
    return render_template('statistiques.html', meta=get_meta(), results=results)


@app.route('/history')
def history_page():
    predictions = database.get_predictions()
    return render_template('historique.html', predictions=predictions)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        message = data.get('message', '')
        if not message.strip():
            return jsonify({'error': 'Veuillez entrer un message.'}), 400

        result = predict(message)

        subject = message.strip().replace('\n', ' ')[:40]
        if len(subject) >= 40:
            subject += '...'
        database.add_prediction(subject, result['label'], result['confidence'], message)

        steps = pipeline_steps(message)
        result['steps'] = [{'title': t, 'text': s} for t, s in steps]
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': 'Erreur lors de la prediction : ' + str(exc)}), 500


@app.route('/api/train', methods=['POST'])
def api_train():
    data = request.get_json(force=True)
    algo = data.get('algo', 'naive_bayes')
    try:
        model, metrics, cm = eda_data.train_algorithm(algo)
        reload_models()
        return jsonify({
            'success': True,
            'algo': algo,
            'model_name': eda_data.ALGO_NAMES[algo],
            'metrics': metrics,
            'confusion_matrix': cm,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/history', methods=['GET'])
def api_history():
    return jsonify(database.get_predictions())


@app.route('/api/current-model', methods=['GET'])
def api_current_model():
    m = get_meta()
    return jsonify({
        'model_name': m.get('current_model', m.get('best_model_name')),
        'accuracy': m.get('current_accuracy', m.get('best_accuracy')),
    })


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
