import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.common import read_yaml
from src.utils.model_utils import load_processed_data, save_model_and_metrics
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
# imbalance handling
from imblearn.over_sampling import SMOTE


# drop this model so forget about it kept here for reference
def train_champion_conversion_model():
  
    try:
        #  Load Configs and Preprocessed Data
        config = read_yaml(Path("configs/config.yaml"))
        params = read_yaml(Path("configs/params.yaml"))
        task_name = "conversion_prediction"
        target_col = params.tasks[task_name].target.name

        logger.info(f"--- Training CHAMPION model for task: {task_name} ---")
        train_df, test_df = load_processed_data(task_name)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # 2. Apply SMOTE ONLY to the Training Data
        logger.info(f"Original training data class distribution: \n{y_train.value_counts()}")
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"Resampled (SMOTE) training data class distribution: \n{y_train_resampled.value_counts()}")

        #L
        logger.info("Initializing XGBClassifier with final champion parameters from params.yaml.")
        champion_params = params.tasks[task_name].model_params_grid.XGBClassifier
        
        champion_model = XGBClassifier(**champion_params)
        
        logger.info("Training the final model on the SMOTE-balanced data...")
        champion_model.fit(X_train_resampled, y_train_resampled)
        logger.info("Training complete.")

        logger.info("Evaluating the model on the original, imbalanced test set...")
        y_pred = champion_model.predict(X_test)
        
        print("\n" + "="*50)
        print("FINAL CHAMPION MODEL REPORT")
        print("="*50)
        print(classification_report(y_test, y_pred))
        
        final_metrics = {
            "model_name": "XGBClassifier_with_SMOTE",
            "final_params": champion_params,
            "test_f1_score_macro": f1_score(y_test, y_pred, average='macro'),
            "test_roc_auc": roc_auc_score(y_test, champion_model.predict_proba(X_test)[:, 1])
        }
        save_model_and_metrics(champion_model, final_metrics, "conversion_prediction_champion")
      #   print(final_metrics) 
    except Exception as e:
        logger.error(f"Error during champion conversion model training: {e}")
        raise e

if __name__ == '__main__':
    train_champion_conversion_model()