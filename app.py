from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)


try:
    model = pickle.load(open("fraud_detection_model.pkl", "rb"))
except FileNotFoundError:
    model = None
    print("Error: fraud_detection_model.pkl not found!")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/dashboard')
def dashboard():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template("index.html", prediction_text="Error: Model file missing.")

    try:
        
        age = float(request.form['age'])
        months = float(request.form['months_as_customer'])
        deduct = float(request.form['policy_deductable'])
        claim = float(request.form['total_claim_amount'])

        features = [[age, months, deduct, claim]]
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        fraud_prob = round(probability[0][1] * 100, 2)

        if claim > 50000 and months < 12:
            result = "High Risk: Fraud Detected"
            fraud_prob = 95.0
            status = "danger" 
        elif prediction[0] == 1 or fraud_prob > 50:
            result = "Fraud Detected"
            status = "danger"
        else:
            result = "No Fraud Detected"
            status = "success"

        return render_template(
            "index.html",
            prediction_text=result,
            fraud_probability=fraud_prob,
            status=status
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)