import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts' ,"train_data.csv")
    test_data_path = os.path.join('artifacts' , "test_data.csv")
    raw_data_path = os.path.join('artifacts' , "data.csv")


class DataIngestion:
    def __init__(self):
        self.Ingestion_config = DataIngestionConfig()


    def initiate_data_Ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            data = pd.read_csv('D:\\DataScienceProject\\end_to_end_project\\notebook\\data\\StudentsPerformance.csv')
            logging.info('read the dataset as datafame')
            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path) , exist_ok=True)

            data.to_csv(self.Ingestion_config.raw_data_path , index = False , header = True)

            logging.info("Train test split initiated")
            train_set ,test_set =train_test_split(data ,train_size=0.2 , random_state=42)

            train_set.to_csv(self.Ingestion_config.train_data_path ,index =False , header = True)

            test_set.to_csv(self.Ingestion_config.test_data_path ,index=False ,header =True)

            logging.info("Igestion of the data is completed")

            return(
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__== "__main__":
    obj=DataIngestion()
    obj.initiate_data_Ingestion()


