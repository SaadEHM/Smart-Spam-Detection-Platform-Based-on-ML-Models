"""Gestion de la base SQLite pour l'historique des predictions."""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'spam_history.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            subject TEXT NOT NULL,
            label TEXT NOT NULL,
            probability REAL NOT NULL,
            message TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def add_prediction(subject, label, probability, message):
    conn = get_connection()
    now = datetime.now().strftime('%d/%m/%Y %H:%M')
    conn.execute(
        'INSERT INTO predictions (date, subject, label, probability, message) VALUES (?, ?, ?, ?, ?)',
        (now, subject, label, probability, message)
    )
    conn.commit()
    conn.close()


def get_predictions(limit=50):
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_predictions():
    conn = get_connection()
    total = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    spam = conn.execute("SELECT COUNT(*) FROM predictions WHERE label='spam'").fetchone()[0]
    conn.close()
    return total, spam
