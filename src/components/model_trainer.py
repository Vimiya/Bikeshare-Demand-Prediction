import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Bagging Regressor": BaggingRegressor(),
                "KNeighbors Regressor":KNeighborsRegressor()

            }

            # I am only mentioning best parameters i got for each model (using Gridsearchcv) while testing on EDA Jupyter notebook present in notebook folder.
            # Remaining parameter values i tested on will be kept commented
            params={
                # "Decision Tree": {
                #     'criterion':['squared_error', 'friedman_mse', 'absolute_error'],
                #     'splitter':['best','random'],
                #     'max_depth':[3,5,7,10,15,20,30,50],
                #     'min_samples_leaf':[3,5,10,15,20,23,25],
                #     'min_samples_split':[8,10,12,18,20],
                #     'max_leaf_nodes':[None,10,20,30,40,50,60]
                # },
 
              "Decision Tree": {
                    'criterion':'friedman_mse',
                    'splitter':'random',
                    'max_depth':30,
                    'min_samples_leaf':3,
                    'min_samples_split':12,
                    'max_leaf_nodes':None
                },

                # "Random Forest":{
                #     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],                 
                #     'max_features':['sqrt','log2',None],
                #     'n_estimators': [8,16,32,64,128,256]
                # },

                "Random Forest":{
                    'criterion':'friedman_mse',                 
                    'max_features':None,
                    'n_estimators': 256
                },

                # "Gradient Boosting":{
                #     'learning_rate': [0.2,0.02,0.02,1],
                #     'max_depth'    : [2,4,6,8,10]
                # },

                "Gradient Boosting":{
                    'learning_rate': 0.2,
                    'max_depth'    : 8
                },


                "Linear Regression":{},


                # "CatBoosting Regressor":{
                #     "iterations": [1000],
                #     "learning_rate": [1e-3, 0.1],
                #     "depth": [1, 10],
                #     "subsample": [0.05, 1.0],
                #     "colsample_bylevel": [0.05, 1.0],
                #     "min_data_in_leaf": [1, 100]
                # },

                "CatBoosting Regressor":{
                    "iterations": 1000,
                    "learning_rate":0.1,
                    "depth": 10,
                    "subsample":1.0,
                    "colsample_bylevel": 1.0,
                    "min_data_in_leaf":1
                },


                # "Bagging Regressor":{
                #     'base_estimator': [None, LinearRegression(), KNeighborsRegressor()],
                #     'n_estimators': [20,50,100],
                #     'max_samples': [0.5,1.0],
                #     'max_features': [0.5,1.0],
                #     'bootstrap': [True, False],
                #     'bootstrap_features': [True, False]},

                "Bagging Regressor":{
                    'base_estimator':None,
                    'n_estimators': 100,
                    'max_samples': 1.0,
                    'max_features': 1.0,
                    'bootstrap':True,
                    'bootstrap_features': False},

                # "KNeighbors Regressor":{
                #     'n_neighbors': [2,3,4,5,6],
                #     'weights': ['uniform','distance']
                # }

                "KNeighbors Regressor":{
                    'n_neighbors': 6,
                    'weights': 'distance'
                }
                
            }

            logging.info("Model evaluation stage and Hyperparameter tuning")

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
                         

            # model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            logging.info(f"Model Evaluation Report: {model_report}") 
            ## To get best model score from dict

            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # putting a threshold that if best model score is not more than 60 percent then raise exception
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # load preprocessor.pkl file to do transformation for the new data that will come in the future for prediction
            # however in this case currently we don't require to load the preprocessor.pkl file 

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model 
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return 'Best Model is ',best_model,' and its score is ',r2_square


        except Exception as e:
            raise CustomException(e,sys)


