import os
import sys
from dataclasses import dataclass

#  These All Algorithm we want to try
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor ,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression , Ridge ,Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_models

import pandas as pd
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_array , test_array):
        try:
            logging.info("split training and test input data")

            train_df =pd.DataFrame(train_array)
            test_df = pd.DataFrame(test_array)

            if train_df.isnull().sum().any() or test_df.isnull().sum().any():
                print("Missing values detected. Proceeding with imputation...")

                train_df.fillna(train_df.median(), inplace=True)
                test_df.fillna(test_df.median(), inplace=True)

                if np.isinf(train_df.values).any() or np.isinf(test_df.values).any():
                    print("Infinite values detected. Replacing with large values...")
    
            # Replace infinite values with a large value (you can choose an appropriate value)
                    train_df.replace([np.inf, -np.inf], 1e10, inplace=True)
                    test_df.replace([np.inf, -np.inf], 1e10, inplace=True)

            # Convert the cleaned pandas DataFrame back to numpy array for model training
                    train_array = train_df.values
                    test_array = test_df.values

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[: ,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
        
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "Linear Regression":LinearRegression(),
                # "Ridge":Ridge(),
                # "Lasso":Lasso(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                # "SVR":SVR(),
                # "KNeighbors" : KNeighborsRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor()
            }

            

            params={

                 "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                 "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Linear Regression":{},
                 "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }


            model_report:dict =evaluate_models(
                X_train=X_train,y_train=y_train,
                X_test=X_test,y_test=y_test , 
                models=models,
                param=params
            ) 
            # To get best model score from dic
            best_model_score = max(sorted(model_report.values()))

            ## To Get best model name fom dic

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            model_r2_score  = r2_score(y_test , predicted)
            return model_r2_score 
        

        except Exception as e:
            raise CustomException(e , sys)