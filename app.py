# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import sklearn

# Load the Random Forest CLassifier model
filename = 'Heart_Disease_RFC.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        Age = int(request.form['Age'])
        Gender = request.form.get('Gender')
        ChestPainType = request.form.get('ChestPainType')
        RestingBP = int(request.form['RestingBP'])
        Cholesterol = int(request.form['Cholesterol'])
        FastingBS = request.form.get('FastingBS')
        RestingECG = int(request.form['RestingECG'])
        MaxHR = int(request.form['MaxHR'])
        ExerciseAngina = request.form.get('ExerciseAngina')
        Oldpeak = float(request.form['Oldpeak'])
        ST_Slope = request.form.get('ST_Slope')
        
        data = np.array([[Age,Gender,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)

