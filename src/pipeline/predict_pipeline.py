import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


# def find_mean_std():
#     try:
#         train_df=pd.read_csv('artifacts/train.csv')
#         numerical_columns=["temp", "hum","windspeed"]
#         train_df=train_df[numerical_columns]
#         mean_df=train_df.mean(axis=0)
#         sd_df=train_df.std(axis=0)
#         return mean_df,sd_df
#     except Exception as e:
#         raise CustomException(e,sys)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            
            
            # since input features are in scaled format,  preds value is also in scaled format
            # so convert the scaled predicted values to normal values            
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        




class CustomData:
    def __init__(self,
        season: str,
        yr: str,
        mnth: str,
        hr: str,
        # holiday: str,
        weekday: str,
        workingday: str,
        weathersit: str,
        temp: float,
        hum: float,
        windspeed: float):

        self.season = season

        self.yr = yr

        self.mnth = mnth

        self.hr = hr

        # self.holiday = holiday

        self.weekday = weekday

        self.workingday = workingday

        self.weathersit = weathersit

        self.temp = temp

        self.hum = hum

        self.windspeed = windspeed

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "season": [self.season],
                "yr": [self.yr],
                "mnth": [self.mnth],
                "hr": [self.hr],
                # "holiday": [self.holiday],
                "weekday": [self.weekday],
                "workingday": [self.workingday],
                "weathersit": [self.weathersit],
                "temp": [self.temp],
                "hum": [self.hum],
                "windspeed": [self.windspeed]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
