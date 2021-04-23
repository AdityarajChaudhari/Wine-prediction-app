import numpy as np
import pickle
import pandas
from flask import Flask,render_template,request,jsonify
from flask_cors import CORS,cross_origin

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@cross_origin()
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@cross_origin()
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        fixed_acidity = float(request.form['Fixed Acidity'])
        volatile_acidity = float(request.form["Volatile Acidity"])
        citric_acid = float(request.form['Citric Acid'])
        residual_sugar = float(request.form["Residual Sugar"])
        chlorides = float(request.form['Chlorides'])
        free_sulfur_dioxide = float(request.form["Free Sulfur Dioxide"])
        total_sulfur_dioxide = float(request.form['Total Sulfur Dioxide'])
        density = float(request.form["Density"])
        pH = float(request.form['pH'])
        sulphate = float(request.form["Sulphates"])
        alcohol = float(request.form['Alcohol'])
        features = [[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphate,alcohol]]
        prediction = model.predict(features)
        output = prediction
        print(output)
        if output == 1:
            return render_template('index.html', prediction_text="Wine is of Awesome Quality")
        else:
            return render_template('index.html', prediction_text="Wine is of Average Quality")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()
