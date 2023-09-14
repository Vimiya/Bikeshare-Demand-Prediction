from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            season=request.form.get('season'),
            yr=request.form.get('yr'),
            mnth=request.form.get('mnth'),
            hr=request.form.get('hr'),
            holiday=request.form.get('holiday'),
            weekday=request.form.get('weekday'),
            workingday=request.form.get('workingday'),
            weathersit=request.form.get('weathersit'),
            temp=float(request.form.get('temp')),
            hum=float(request.form.get('hum')),
            windspeed=float(request.form.get('windspeed'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        

