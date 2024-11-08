import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path ,exist_ok = True)
        with open(file_path , 'wb') as file:
            dill.dump(obj , file)

    except Exception as e:
        raise CustomException(sys,e)