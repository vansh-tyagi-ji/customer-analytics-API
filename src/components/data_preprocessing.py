import os
import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.common import read_yaml, create_directories
import joblib

# Import all the powerful tools we need
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# hah tried my best to keep it modular and clean and add comments wherever possible
class DataPreprocessing:
    def __init__(self, config_path="configs/config.yaml", params_path="configs/params.yaml"):
        logger.info("Data Preprocessing component initialized.")
        self.config = read_yaml(Path(config_path))
        self.params = read_yaml(Path(params_path))
        create_directories([self.config.data_preprocessing.root_dir, self.config.model_trainer.trained_models_dir])

    def preprocess_task(self, task_name: str):
        """preprocesses data for a specific task defined in params.yaml (e.g- 'churn_prediction')."""
        try:
            task_config = self.params.tasks[task_name]
            logger.info(f" Starting preprocessing for task: {task_name} ")

                # Load the ingested data
            task_dir = Path(os.path.join(self.config.data_preprocessing.root_dir, task_name))
            create_directories([task_dir])
            ingestion_root = self.config.data_ingestion.root_dir
            # check if split was done during ingestion
            if task_config.split_required:
                train_path = Path(os.path.join(ingestion_root, task_config.train_data_filename))
                test_path = Path(os.path.join(ingestion_root, task_config.test_data_filename))
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                logger.info(f"Loaded train ({train_df.shape}) and test ({test_df.shape}) data for {task_name}.")
            # if not because kmean task didnt require split
            else:
                full_path = Path(os.path.join(ingestion_root, task_config.output_filename))
                train_df = pd.read_csv(full_path)
                test_df = None
                logger.info(f"Loaded full dataset ({train_df.shape}) for {task_name}.")

           
            if task_config.target and task_config.target.name:
                target_col = task_config.target.name
                X_train = train_df.drop(columns=[target_col])
                y_train = train_df[target_col]
                X_test = test_df.drop(columns=[target_col]) if test_df is not None else None
                y_test = test_df[target_col] if test_df is not None else None
            # segmentation task case (hmm all else case are mostly for segmentation)    
            else:
                X_train, y_train, X_test, y_test = train_df, None, test_df, None

            
            num_features = task_config.get("numerical_features", [])
            cat_features = task_config.get("categorical_features", [])
            text_features = task_config.get("text_features", [])
            
            # data cleaning: handle missing values
            for col in text_features:
                X_train[col].fillna("", inplace=True)
                if X_test is not None:
                    X_test[col].fillna("", inplace=True)

            
            numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
            categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                    ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=0.01, sparse_output=False))])
            # did max_features=500 to limit dimensionality explosion from text
            text_pipelines = [(f'text_{col}', TfidfVectorizer(max_features=500), col) for col in text_features]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_pipeline, num_features),
                    ('cat', categorical_pipeline, cat_features)
                ] + text_pipelines,
                remainder='drop'
            )

            # Fit on train data and transform both train and test data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test) if X_test is not None else None

            # for this part took ai help :)
            new_cols = preprocessor.get_feature_names_out()
            X_train_processed_df = pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed, columns=new_cols)
            X_test_processed_df = pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed, columns=new_cols) if X_test_processed is not None else None
            
            if y_train is not None:
                train_final = pd.concat([X_train_processed_df, y_train.reset_index(drop=True)], axis=1)
                if X_test_processed_df is not None:
                    test_final = pd.concat([X_test_processed_df, y_test.reset_index(drop=True)], axis=1)
        
        # save the processed data
            if task_config.split_required:           
                train_final.to_csv(task_dir / "train_processed.csv", index=False)
            else:
                X_train_processed_df.to_csv(task_dir / "train_processed.csv", index=False)    
            logger.info(f"Processed train data saved for {task_name}")
            if X_test_processed_df is not None:
                test_final.to_csv(task_dir / "test_processed.csv", index=False)
                logger.info(f"Processed test data saved for {task_name}")

            # save the preprocessor object
            preprocessor_path = Path(self.config.model_trainer.trained_models_dir) / f"{task_name}_preprocessor.pkl"
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessor for {task_name} saved to {preprocessor_path}")

        except Exception as e:
            logger.error(f"Error during preprocessing for task '{task_name}': {e}")
            raise e