import os
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# (field name, label, min, max)
# ORDER wahi hona chahiye jo football.py me FEATURES ka hai
FIELDS = [
    ('age', 'Age', 15, 50),
    ('pace', 'Pace', 1, 99),
    ('shooting', 'Shooting', 1, 99),
    ('passing', 'Passing', 1, 99),
    ('dribbling', 'Dribbling', 1, 99),
    ('defending', 'Defending', 1, 99),
    ('physic', 'Physical', 1, 99),
]

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        "model.pkl not found. Pehle 'python football.py' chalao taa ke model.pkl ban jaye."
    )


def render(**kwargs):
    return render_template('index.html', fields=FIELDS, **kwargs)


@app.route('/')
def index():
    return render(values={}, result=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    values = {name: request.form.get(name, '').strip() for name, *_ in FIELDS}
    features = []

    for name, label, low, high in FIELDS:
        raw = values[name]
        if raw == '':
            return render(values=values, result=None, error=f'{label} is required.')
        try:
            number = float(raw)
        except ValueError:
            return render(values=values, result=None, error=f'{label} must be a number.')
        if not low <= number <= high:
            return render(values=values, result=None,
                          error=f'{label} must be between {low} and {high}.')
        features.append(number)

    arr = np.array([features], dtype=np.float64)
    pred = float(model.predict(arr)[0])
    rating = int(round(min(max(pred, 1), 99)))  # rating 1-99 ke beech rakho
    return render(values=values, result=rating, error=None)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, port=port)
