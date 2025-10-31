import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.common import read_yaml
# Import necessary functions from your utils
from src.utils.model_utils import load_processed_data, save_model_and_metrics 

# Import the champion model and metrics
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# in this i find it boring to add comments every where as i have done in other files so please ignore lack of comments here
def train_champion_clv_model():
    """
    Trains the single best CLV model (XGBoost) using the optimal hyperparameters
    read directly from params.yaml.
    """
    try:
        # Load Configs and Preprocessed Data
        config = read_yaml(Path("configs/config.yaml"))
        params = read_yaml(Path("configs/params.yaml"))
        task_name = "clv_prediction"
        target_col = params.tasks[task_name].target.name

        logger.info(f"--- Training CHAMPION model for task: {task_name} ---")
        
        train_df, test_df = load_processed_data(task_name) # Pass config here

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]


        logger.info("Initializing XGBRegressor with final champion parameters from params.yaml.")
        
        champion_params_dict = params.tasks[task_name].model_params_grid.XGBRegressor
        champion_model = XGBRegressor(**champion_params_dict)
        
        logger.info("Training the final model...")
        champion_model.fit(X_train, y_train)
        logger.info("Training complete.")

        logger.info("Calculating feature importances...")
        feature_names = X_train.columns

        importances = champion_model.feature_importances_

        importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        print("\n" + "="*50)
        print("FEATURE IMPORTANCES (Top 15)")
        print("="*50)
        print(importance_df.head(30).to_string(index=False)) 
        print("="*50 + "\n")

        y_pred = champion_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print("\n" + "="*50)
        print("FINAL CHAMPION MODEL REPORT")
        print("="*50)
        print(f"  R-squared (R²): {r2:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print("="*50)
        
        # Save the final model and metrics
        final_metrics = {
            "model_name": "XGBRegressor",
            "final_params": champion_params_dict, 
            "r2_score": r2, 
            "rmse": rmse
        }
        # Save model with a clear name, pass config
        save_model_and_metrics(champion_model, final_metrics, "clv_prediction_champion") 

    except Exception as e:
        logger.error(f"Error during champion CLV model training: {e}")
        raise e

if __name__ == '__main__':
    train_champion_clv_model()