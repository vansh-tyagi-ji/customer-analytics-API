import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.common import read_yaml, create_directories

class DataIngestion:
    def __init__(self, config_path="configs/config.yaml", params_path="configs/params.yaml"):
        logger.info("Data Ingestion started.")
        self.config = read_yaml(Path(config_path))
        self.params = read_yaml(Path(params_path))
        
        create_directories([self.config.data_ingestion.root_dir])

        # personal note:
        #  well iski jarurat nhi hai abb but rakh dete hain kyuki recommendation hata diya toh order file ki jarurat nhi rahi customer se hi ho jayega
        source_data_dir = self.config["data_preperation"]["root_dir"] 
        self.source_paths = {
            'customer_file': Path(os.path.join(source_data_dir, self.params.data_ingestion.source_customer_filename)),
            'order_file': Path(os.path.join(source_data_dir, self.params.data_ingestion.source_order_filename))
        }
        logger.info("Configuration and parameters loaded perfectly.")

    def process_task(self, task_name: str):
        """
        Processes a specific task defined in params.yaml (e.g., 'churn_prediction').
        """
        logger.info(f" Starting data ingestion for task: {task_name}        ")
        
        try:
            task_config = self.params.tasks[task_name]
            
            # load data
            source_file_key = task_config.source_data
            data_path = self.source_paths[source_file_key]
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path} with shape: {df.shape}")

            # target column creation (if any)
            target_name = None
            if task_config.get('target') and task_config.target.get('name'):
                target_name = task_config.target.name
                if task_config.target.get('creation_logic'):
                    creation_logic = task_config.target.creation_logic
                    df[target_name] = df.eval(creation_logic).astype(int)
                    logger.info(f"Target column '{target_name}' created successfully.")
                elif target_name not in df.columns:
                    raise ValueError(f"Target column '{target_name}' not in data and no creation_logic provided.")

            # Feature Selection
            num_features = task_config.get("numerical_features", [])
            cat_features = task_config.get("categorical_features", [])
            text_features = task_config.get("text_features", [])
            
            # Combine all feature lists into one
            all_features = num_features + cat_features + text_features
            
            # check if features are defined
            if not all_features:
                raise ValueError(f"No features defined for task: {task_name}")

            logger.info(f"Selecting {len(all_features)} features for the task '{task_name}'.")
            
            columns_to_keep = all_features + ([target_name] if target_name else [])
            
            #  simple checking for missing columns
            missing_cols = [col for col in columns_to_keep if col not in df.columns]
            if missing_cols:
                raise ValueError(f"The following columns are missing in the dataframe for task '{task_name}': {missing_cols}")

            final_df = df[columns_to_keep]

            # Perform train/test split if required (😊not required for segmentation)
            output_dir = Path(self.config.data_ingestion.root_dir)
            if task_config.split_required:
                train_data, test_data = train_test_split(
                    final_df,
                    test_size=self.params.training.test_size,
                    random_state=self.params.training.random_state,
                    stratify=final_df[target_name] if target_name and final_df[target_name].nunique() < 20 else None
                )
                
                train_path = output_dir / task_config.train_data_filename
                test_path = output_dir / task_config.test_data_filename
                
                # Save the datasets
                train_data.to_csv(train_path, index=False)
                test_data.to_csv(test_path, index=False)
                logger.info(f"Train data saved to: {train_path}")
                logger.info(f"Test data saved to: {test_path}")

            else: # If no split is needed (like for segmentation)
                output_path = output_dir / task_config.output_filename
                final_df.to_csv(output_path, index=False)
                logger.info(f"Full dataset for task saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error processing task '{task_name}': {e}")
            raise e