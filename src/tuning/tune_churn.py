# src/tuning/tune_churn_simple.py

import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.common import read_yaml

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# this tuning is just another testing way ignore it
# i did tunning in model_tuning notebook 
def tune_churn_simple():
    try:
       
        config = read_yaml(Path("configs/config.yaml"))
        params = read_yaml(Path("configs/params.yaml"))
        task_name = "churn_prediction"
        
        
        raw_data_path = "data/raw/" + params['data_ingestion']['source_customer_filename']
        df = pd.read_csv(raw_data_path)
        df.dropna(subset=['customer_tenure_days', 'days_since_last_order'], inplace=True)

        
        df['churn'] = (df['days_since_last_order'] > 180).astype(int)
        
        
        y = df['churn']
        
        num_features = params['tasks'][task_name]['numerical_features']
        cat_features = params['tasks'][task_name]['categorical_features']
        
        
        if 'days_since_last_order' in num_features:
            num_features.remove('days_since_last_order')
            
        X = df[num_features + cat_features]

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=params['training']['test_size'], 
            random_state=params['training']['random_state'],
            stratify=y
        )

        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ])

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        
        models = {
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "XGBClassifier": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        
        model_params_grid = params['tasks'][task_name]['model_params_grid']
        
        for model_name, model in models.items():
            logger.info(f"--- Tuning {model_name} ---")
            param_grid = model_params_grid[model_name]
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_processed, y_train)
            
            best_model = grid_search.best_estimator_
            
            print("\n" + "="*30)
            print(f"MODEL: {model_name}")
            print(f"Best CV Score (ROC AUC): {grid_search.best_score_:.4f}")
            print("Best Parameters:", grid_search.best_params_)
            
            y_pred = best_model.predict(X_test_processed)
            print("\nTest Set Report:")
            print(classification_report(y_test, y_pred))
            print("="*30 + "\n")

    except Exception as e:
        logger.error(f"Error in churn tuning: {e}")
        raise e

if __name__ == '__main__':
    tune_churn_simple()