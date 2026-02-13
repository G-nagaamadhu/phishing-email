from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["email"]
    vector = vectorizer.transform([data])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][1]

    return jsonify({
        "result": "Phishing" if prediction == 1 else "Legitimate",
        "risk_score": round(probability * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)