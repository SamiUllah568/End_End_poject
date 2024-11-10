import os
import sys
from src.exception import CustomException
import dill
from src.logger import logging
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path ,exist_ok = True)
        with open(file_path , 'wb') as file:
            dill.dump(obj , file)

    except Exception as e:
        raise CustomException(sys,e)




def evaluate_models(X_train,y_train,X_test,y_test , models):
    try:
        report={}
        for model_name, model_class in models.items():
            model = model_class

            model.fit(X_train , y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train , y_train_pred)
            test_model_score = r2_score(y_test , y_test_pred)

            logging.info(f"{model_name} - Train R^2 Score: {train_model_score}, Test R^2 Score: {test_model_score}")

            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)