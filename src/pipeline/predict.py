# In src/pipeline/predict.py
import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.common import read_yaml
from src.utils.model_utils import load_artefacts

# # forget this because recommendation and pricing is removed
class PredictionPipeline:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.config = read_yaml(Path("configs/config.yaml"))
        self.params = read_yaml(Path("configs/params.yaml"))
        
        self.model, self.preprocessor = load_artefacts(task_name, self.config)
        
        self.num_features = self.params['tasks'][task_name].get("numerical_features", [])
        self.cat_features = self.params['tasks'][task_name].get("categorical_features", [])
        self.text_features = self.params['tasks'][task_name].get("text_features", [])
        self.features = self.num_features + self.cat_features + self.text_features
        
    def predict(self, data):
        """
        """
        try:
            data_df = pd.DataFrame(data)
            data_df = data_df[self.features]

            processed_data = self.preprocessor.transform(data_df)
            
            prediction = self.model.predict(processed_data)

            if hasattr(self.model, 'predict_proba'):
                positive_class_probability = self.model.predict_proba(processed_data)[:, 1]
                return {"prediction": prediction.tolist(), "positive_class_probability": positive_class_probability.tolist()}

            return {"prediction": prediction.tolist()}

        except Exception as e:
            logger.error(f"Error during prediction for task '{self.task_name}': {e}")
            raise e