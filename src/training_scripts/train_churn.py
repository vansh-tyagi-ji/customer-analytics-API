import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import logger
from src.utils.common import read_yaml
# Import necessary functions from your utils
from src.utils.model_utils import load_processed_data, save_model_and_metrics 

# Import the champion model and metrics
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

def train_champion_churn_model():
    try:
        # Load Configs and Preprocessed Data
        config = read_yaml(Path("configs/config.yaml"))
        params = read_yaml(Path("configs/params.yaml"))
        task_name = "churn_prediction"
        target_col = params.tasks[task_name].target.name

        logger.info(f"--- Training CHAMPION model for task: {task_name} ---")
        train_df, test_df = load_processed_data(task_name)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # tried tunning then this is the best params found so far 
        # thats why called champion model
        # 1. Initialize the model with champion parameters from params.yaml
        logger.info("Initializing XGBClassifier with final champion parameters from params.yaml.")
        champion_params = params.tasks[task_name].model_params_grid.XGBClassifier
        champion_model = XGBClassifier(**champion_params) # use this kwargs 
        
        # Train the model
        logger.info("Training the final model...")
        champion_model.fit(X_train, y_train)
        logger.info("Training complete.")

        # from here onward code is little messy but works for reporting purpose try many time and i tink now even i forget what is done here
        logger.info("Calculating feature importances")
        feature_names = X_train.columns
        importances = champion_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print("\n" + "="*50)
        print("FEATURE IMPORTANCES (Top 15)")
        print("="*50)
        print(importance_df.head(15).to_string(index=False))
        print("="*50 + "\n")

        # 5. Evaluate at Multiple Thresholds
        logger.info("Evaluating model performance at different thresholds...")
        y_pred_proba = champion_model.predict_proba(X_test)[:, 1]
        
        thresholds_to_test = [0.68,0.63, 0.7,0.72,0.66, 0.73] # Define thresholds to check
        
        print("\n" + "="*60)
        print("PERFORMANCE AT DIFFERENT PROBABILITY THRESHOLDS")
        print("="*60)
        
        for threshold in thresholds_to_test:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            print(f"\nClassification Report for Threshold = {threshold:.2f} ")
            print(classification_report(y_test, y_pred_threshold))
            print("-"*60)

        # optimal threshold from params.yaml find during tunnig and this time as well
        optimal_threshold = params.tasks[task_name].optimal_threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        print("\n" + "="*50)
        print("FINAL REPORT (Using Optimal Threshold from params.yaml)")
        print(f"(Optimal Threshold = {optimal_threshold:.2f})")
        print("="*50)
        print(classification_report(y_test, y_pred_optimal))
        
    #    save the champion model and final metrics
        final_metrics = {
            "model_name": "XGBClassifier",
            "final_params": champion_params,
            "optimal_threshold": optimal_threshold,
            "test_f1_score_macro": f1_score(y_test, y_pred_optimal, average='macro'), 
            "test_roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
       
        save_model_and_metrics(champion_model, final_metrics, "churn_prediction_champion") 
        logger.info("Champion model and metrics saved.")

    except Exception as e:
        logger.error(f"Error during champion churn model training: {e}")
        raise e

if __name__ == '__main__':
    train_champion_churn_model()