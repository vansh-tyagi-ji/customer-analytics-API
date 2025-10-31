# src/utils/model_utils.py

import pandas as pd
import joblib
import json
from pathlib import Path
from src.utils.logger import logger
from src.utils.common import read_yaml, create_directories
from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_squared_error
import numpy as np
import pandas as pd

def load_processed_data(task_name: str):
    """Loads the preprocessed train and test data for a given task."""
    try:
        config = read_yaml(Path("configs/config.yaml"))
        preprocessed_dir = Path(config['data_preprocessing']['root_dir']) / task_name
        train_path = preprocessed_dir / "train_processed.csv"
        test_path = preprocessed_dir / "test_processed.csv"
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loaded train ({train_df.shape}) and test ({test_df.shape}) data for {task_name}")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading data for task {task_name}: {e}")
        raise e

def save_model_and_metrics(model, metrics: dict, task_name: str):
    """Saves the trained model and its performance metrics."""
    try:
        # Save the model
        config = read_yaml(Path("configs/config.yaml"))
        models_dir = Path(config['model_trainer']['trained_models_dir'])
        create_directories([models_dir])

        model_path = models_dir / f"{task_name}_model.pkl"
        joblib.dump(model, model_path)
        
        # Save the metrics
        metrics_path = models_dir / f"{task_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Model for '{task_name}' saved to {model_path}")
        logger.info(f"Metrics for '{task_name}' saved to {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving model or metrics for {task_name}: {e}")
        raise e
    
# src/utils/model_utils.py

# ... (keep all your existing functions) ...
# import joblib # Make sure joblib is imported at the top

# def load_artefacts(task_name: str, config: dict):
#     """Loads the saved model and scaler for a given task."""
#     try:
#         models_dir = Path(config['model_trainer']['trained_models_dir'])
        
#         # Load the model
#         model_path = models_dir / f"{task_name}_model.pkl"
#         model = joblib.load(model_path)
        
#         # Load the scaler (saved during preprocessing)
#         # Note: The scaler path from your preprocessing script needs to be correct.
#         # Assuming scalers are also in the 'models' directory.
#         scaler_path = models_dir / f"{task_name}_scaler.pkl" 
#         scaler = joblib.load(scaler_path)
        
#         logger.info(f"Loaded model and scaler for task: {task_name}")
#         return model, scaler
#     except Exception as e:
#         logger.error(f"Error loading artefacts for task {task_name}: {e}")
#         raise e

# In src/utils/model_utils.py

def load_artefacts(task_name: str, config: dict):
    """Loads the saved model and the corresponding preprocessor for a given task."""
    try:
        models_dir = Path(config['model_trainer']['trained_models_dir'])
        
        # Load the model (this part stays the same)
        model_path = models_dir / f"{task_name}_champion_model.pkl"
        model = joblib.load(model_path)
        
        # CHANGE: Load the preprocessor instead of the scaler
        preprocessor_path = models_dir / f"{task_name}_preprocessor.pkl" 
        preprocessor = joblib.load(preprocessor_path)
        
        logger.info(f"Loaded model and preprocessor for task: {task_name}")
        return model, preprocessor # Return the preprocessor object
    except Exception as e:
        logger.error(f"Error loading artefacts for task {task_name}: {e}")
        raise e

def evaluate_regression_model(y_true, y_pred):
    """Calculates regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {"r2_score": r2, "rmse": rmse}
    logger.info(f"Regression Metrics: {metrics}")
    return metrics


def evaluate_classification_model(y_true, y_pred_proba):
    """Calculates classification metrics."""
    # y_pred_proba = y_pred_proba
    auc = roc_auc_score(y_true, y_pred_proba)
    # Use a standard 0.5 threshold for F1-score
    f1 = f1_score(y_true, (y_pred_proba > 0.5).astype(int))
    metrics = {"roc_auc": auc, "f1_score": f1}
    logger.info(f"Classification Metrics: {metrics}")
    return metrics