# src/training_scripts/train_recommender.py

import pandas as pd
from pathlib import Path
import joblib
from src.utils.logger import logger
from src.utils.common import read_yaml

# Imports from the surprise library
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# drop this model so forget about it kept here for reference
def train_recommender_model():
    try:
        task_name = "recommendation_system"
        logger.info(f"Starting training for: {task_name} ")
        config = read_yaml(Path("configs/config.yaml"))
        params = read_yaml(Path("configs/params.yaml"))
        
        order_data_path = Path(config['data_ingestion']['root_dir']) / params['data_ingestion']['source_order_filename']
        df = pd.read_csv(order_data_path)
        
        recommender_df = df[['customer_id', 'product_id', 'review_score']].copy()
        recommender_df.dropna(inplace=True) 
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(recommender_df, reader)
        trainset = data.build_full_trainset()
        model = SVD(n_factors=100, n_epochs=20, random_state=42)
        
        model.fit(trainset)
        logger.info("Model training complete.")
        model_save_path = Path(config['model_trainer']['trained_models_dir']) / f"{task_name}_model.pkl"
        joblib.dump(model, model_save_path)
        logger.info(f"✅ Model for task '{task_name}' saved to: {model_save_path}")

    except Exception as e:
        logger.error(f"Error in recommender model training script: {e}")
        raise e

if __name__ == '__main__':
    train_recommender_model()