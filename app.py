from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load SVM model
model = joblib.load('svm_model.pkl')

@app.route("/", methods=['GET'])
def index():
    return render_template("cek.html")

@app.route("/predict", methods=['POST'])
def predict():
    # mengambil data
    data = request.form.to_dict()
    values = [
        float(data['kosa-kata-buruk']),
        float(data['ingatan-buruk']),
        float(data['kecepatan-membaca-buruk']),
        float(data['klasifikasi-objek-buruk']),
        float(data['identifikasi-suara-buruk'])
    ]
    
    # menghitung survey score
    survey_score = round(sum(values) / 5, 2)

    # Create the features array including the survey score
    features = np.array([[
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        survey_score
    ]])
    
    # Memprediksi
    prediction = model.predict(features)[0]
    
    # Mengubah hasil menjadi teks
    if prediction == 0:
        result_text = "Normal"
    elif prediction == 1:
        result_text = "Surface dyslexia"
    elif prediction == 2:
        result_text = "Deep dyslexia"
    
    return jsonify({
        'survey_score': survey_score,
        'prediction': result_text
    })

if __name__ == '__main__':
    app.run(port=3000, debug=True)
