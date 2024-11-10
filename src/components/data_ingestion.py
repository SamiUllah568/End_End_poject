import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts' ,"train_data.csv")
    test_data_path = os.path.join('artifacts' , "test_data.csv")
    raw_data_path = os.path.join('artifacts' , "data.csv")


class DataIngestion:
    def __init__(self):
        self.Ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            data = pd.read_csv('D:\\DataScienceProject\\end_to_end_project\\notebook\\data\\StudentsPerformance.csv')
            logging.info('read the dataset as datafame')
            
            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path) , exist_ok=True)

            data.to_csv(self.Ingestion_config.raw_data_path , index = False , header = True)

            logging.info("Train test split initiated")
            train_set ,test_set =train_test_split(data ,test_size=0.2 , random_state=42)

            train_set.to_csv(self.Ingestion_config.train_data_path ,index =False , header = True)

            test_set.to_csv(self.Ingestion_config.test_data_path ,index=False ,header =True)

            logging.info("Igestion of the data is completed")

            return(
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(sys,e)
        

if __name__== "__main__":
    obj=DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data , test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr ,test_arr))

