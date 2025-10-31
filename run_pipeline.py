from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.utils.logger import logger

#  Data Ingestion Stage ===
try:
    data_ingestion_component = DataIngestion()

    data_ingestion_component.process_task(task_name="churn_prediction")
    data_ingestion_component.process_task(task_name="clv_prediction")
    data_ingestion_component.process_task(task_name="customer_segmentation")
    # data_ingestion_component.process_task(task_name="conversion_prediction")
    # data_ingestion_component.process_task(task_name="recommendation_system") 
    logger.info(" Stage 1: Data Ingestion Completed Successfully \n\n")

except Exception as e:
    logger.error(f"Data Ingestion Stage Failed: {e}")
    raise e


#  Data Preprocessing Stage 
try:
    logger.info(" Stage 2: Data Preprocessing Started ")
    data_preprocessing_component = DataPreprocessing()

    data_preprocessing_component.preprocess_task(task_name="churn_prediction")
    data_preprocessing_component.preprocess_task(task_name="clv_prediction")
    data_preprocessing_component.preprocess_task(task_name="customer_segmentation")
    # data_preprocessing_component.preprocess_task(task_name="conversion_prediction")
    
    logger.info(" Stage 2: Data Preprocessing Completed Successfully \n\n")


except Exception as e:
    logger.error(f"Data Preprocessing Stage Failed: {e}")
    raise e
