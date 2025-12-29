from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

# Load model & scaler
model, scaler = pickle.load(open("model.pkl", "rb"))

# Load real column names
data = load_breast_cancer()
original_names = data.feature_names  # original feature names

# Convert to HTML-safe names (no spaces)
safe_names = [name.replace(" ", "_") for name in original_names]

@app.route('/')
def home():
    return render_template("index.html", feature_names=safe_names)

@app.route('/predict', methods=['POST'])
def predict():
    values = []
    for name in safe_names:
        values.append(float(request.form[name]))

    final_input = scaler.transform([values])
    pred = model.predict(final_input)[0]

    if pred == 0:
        result = "Breast Cancer Detected (Malignant)"
    else:
        result = "No Breast Cancer (Benign)"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
