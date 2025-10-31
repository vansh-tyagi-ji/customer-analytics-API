import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.common import read_yaml
# Import necessary functions from your utils
from src.utils.model_utils import load_processed_data, save_model_and_metrics

# Import the champion model and metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np # Import numpy for potential type handling in metrics

# here also i will skip adding comments as i have done in other files(train_churn.py) so please ignore lack of comments here
def train_champion_segmentation_model():
    try:
        #  Load Configs and Preprocessed Data
        config = read_yaml(Path("configs/config.yaml"))
        params = read_yaml(Path("configs/params.yaml"))
        task_name = "customer_segmentation"

        logger.info(f"--- Training CHAMPION model for task: {task_name} ---")
        
        preprocessed_dir = Path(config['data_preprocessing']['root_dir']) / task_name
        data_path = preprocessed_dir / "train_processed.csv"
        df = pd.read_csv(data_path)
        logger.info(f"Loaded full dataset ({df.shape}) for {task_name} from {data_path}")
        logger.info("Initializing KMeans with final champion parameters from params.yaml.")
        
        champion_params = params.tasks[task_name].model_params_grid.KMeans
     
        champion_model = KMeans(**champion_params)
        
        logger.info("Training the final model...")
        champion_model.fit(df)
        logger.info("Training complete.")
        logger.info("Evaluating the model using Silhouette Score...")
        final_score = silhouette_score(df, champion_model.labels_)
        
        print("\n" + "="*50)
        print("FINAL CHAMPION MODEL REPORT")
        print("="*50)
        print(f"  Final Silhouette Score: {final_score:.4f}")
        print(f"  Optimal Number of Clusters (k): {champion_params['n_clusters']}")
        print("="*50)
        
        final_metrics = {
            "model_name": "KMeans",
            "final_params": champion_params, 
            "silhouette_score": final_score
        }
        # Save model with a clear name
        save_model_and_metrics(champion_model, final_metrics, "customer_segmentation_champion")

    except Exception as e:
        logger.error(f"Error during champion segmentation model training: {e}")
        raise e

if __name__ == '__main__':
    train_champion_segmentation_model()