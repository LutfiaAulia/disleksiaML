from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the SVM model
model = joblib.load('svm_model.pkl')

@app.route("/", methods=['GET'])
def index():
    return render_template("cek.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.form.to_dict()
    values = [
        float(data['kosa-kata-buruk']),
        float(data['ingatan-buruk']),
        float(data['kecepatan-membaca-buruk']),
        float(data['klasifikasi-objek-buruk']),
        float(data['identifikasi-suara-buruk'])
    ]
    
    # Calculate survey score
    survey_score = sum(values) / 5

    # Create the features array including the survey score
    features = np.array([[
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        survey_score
    ]])
    
    # Make prediction
    prediction = model.predict(features)
    decision_values = model.decision_function(features)
    
    # Convert decision values to probability
    probability = (np.exp(decision_values) / np.sum(np.exp(decision_values), axis=1)).tolist()

    return jsonify({
        'prediction': int(prediction[0]),
        'probability': probability[0]
    })

if __name__ == '__main__':
    app.run(port=3000, debug=True)
