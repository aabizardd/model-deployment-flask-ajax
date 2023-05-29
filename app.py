from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load(open("model_knn.jlb", "rb"))

@app.route("/")
def home():
    return render_template('index.html')


@app.route('/prediksi', methods=["POST"])
def prediksi():
    
    data1 = float(request.form['a'])
    data2 = float(request.form['b']) 
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    
    nama_bunga = ""
    
    if pred[0] == 0:
        nama_bunga = "Setosa"
    elif pred[0] == 1:
        nama_bunga = "Versicolor"
    else:
        nama_bunga = "Virginica"
        
    return render_template('index.html', prediction = "{}".format(nama_bunga))


@app.route('/predict', methods=["POST"])
def predict():
    
    data1 = float(request.form['a'])
    data2 = float(request.form['b']) 
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)

    # Mengembalikan hasil prediksi dalam format JSON
    result = {
        'prediction': pred.tolist()  # Convert prediction to a list
    }
    
    return jsonify(result)
    

if __name__ == "__main__":
    app.run(debug=True)
