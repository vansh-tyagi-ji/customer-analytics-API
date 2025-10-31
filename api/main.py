# main.py
from pydantic import BaseModel 
from datetime import date
import pandas as pd
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, computed_field
from src.utils.logger import logger
from src.utils.common import read_yaml
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)
class CustomerBaseFeatures(BaseModel):
    days_since_last_order: int = 0
    total_orders: int = 0
    total_spent: float = 0.0
    avg_review_score: float = 3.0
    customer_tenure_days: int = 0
    total_items: int = 0
    total_freight: float = 0.0
    customer_state: str = "SP"
    review_speed_category: str = "Normal"
    first_order_date: date = date.today()

    # try to reduce input feild by computed fields 😌
    @computed_field
    @property
    def avg_price(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.total_spent / self.total_items

    @computed_field
    @property
    def avg_items_per_order(self) -> float: 
        if self.total_orders == 0: 
            return 0.0
        return self.total_items / self.total_orders

    @computed_field
    @property
    def tenure_X_spending(self) -> float:
        return self.customer_tenure_days * self.total_spent

    @computed_field
    @property
    def review_X_orders(self) -> float:
        return self.avg_review_score * self.total_orders

    @computed_field
    @property
    def freight_per_order(self) -> float:
        if self.total_orders == 0:
            return 0.0
        return self.total_freight / self.total_orders

    @computed_field
    @property
    def purchase_month(self) -> int:
        return self.first_order_date.month

    @computed_field
    @property
    def purchase_day_of_week(self) -> int:
        return self.first_order_date.weekday()

    @computed_field
    @property
    def purchase_week_of_year(self) -> int:
        return self.first_order_date.isocalendar()[1]

# Pydantic Output Model
class AnalysisResponse(BaseModel):
    segment_id: int
    segment_name: str
    churn_probability: float
    churn_decision: str
    churn_risk_level: str
    predicted_clv: float

app = FastAPI(title="Customer Analytics API")
models = {} # store model and their preprocessors here
params = {}

# Segment Mapping used in ui and response
SEGMENT_MAP = {
    0: "Champions",
    1: "At-Risk Customers",
    2: "Loyal Customers",
    3: "New/Lost Customers"
}

#load models on startup
@app.on_event("startup")
def load_resources():
    global models, params
    config = read_yaml(Path("configs/config.yaml"))
    params = read_yaml(Path("configs/params.yaml"))
    model_dir = Path(config['model_trainer']['trained_models_dir'])

    models["churn_threshold"] = params['tasks']['churn_prediction']['optimal_threshold']
    
    # load models and preprocessors
    try:
        models["segmentation_model"] = joblib.load(model_dir / "customer_segmentation_champion_model.pkl")
        models["segmentation_preprocessor"] = joblib.load(model_dir / "customer_segmentation_preprocessor.pkl")
        
        models["churn_model"] = joblib.load(model_dir / "churn_prediction_champion_model.pkl")
        models["churn_preprocessor"] = joblib.load(model_dir / "churn_prediction_preprocessor.pkl")
        
        models["clv_model"] = joblib.load(model_dir / "clv_prediction_champion_model.pkl")
        models["clv_preprocessor"] = joblib.load(model_dir / "clv_prediction_preprocessor.pkl")
        
        logger.info("All champion models and preprocessors loaded successfully.")
    except Exception as e:
        logger.error(f" ERROR: Could not load models on startup. {e}")
        raise e


# Main API Endpoint

@app.post("/analyze_customer/", response_model=AnalysisResponse)
def analyze_customer(features: CustomerBaseFeatures):
    
    logger.info("Received request for customer analysis...")
    
    # Input DataFrame
    input_data = pd.DataFrame([features.model_dump()])

    try:
        # well segmentation has their own features so we need to compute them first
        seg_params = params['tasks']['customer_segmentation']
        seg_features_list = seg_params['numerical_features'] + seg_params['categorical_features']
        
        seg_input_data = input_data[seg_features_list]

        seg_features_processed = models["segmentation_preprocessor"].transform(seg_input_data)
        segment_id = int(models["segmentation_model"].predict(seg_features_processed)[0])
        
        # Add segment_id to input_data for downstream tasks
        input_data['segment_id'] = segment_id
        segment_name = SEGMENT_MAP.get(segment_id, "Unknown Segment")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation error: {e}")

    # Churn Predict 
    try:
        # churn features 
        churn_params = params['tasks']['churn_prediction']
        churn_features_list = churn_params['numerical_features'] + churn_params['categorical_features']
        # did not add segment_id here as it is present in churn features already
        churn_input_data = input_data[churn_features_list]

        # Prediction
        churn_features_processed = models["churn_preprocessor"].transform(churn_input_data)
        churn_prob = float(models["churn_model"].predict_proba(churn_features_processed)[0, 1])

        threshold = models["churn_threshold"]
        churn_decision = "Churn" if churn_prob >= threshold else "Not Churn"

        if churn_prob >= 0.7:
            churn_risk_level = "High Risk"
        elif churn_prob >= 0.4:
            churn_risk_level = "Medium Risk"
        else:
            churn_risk_level = "Low Risk"
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Churn prediction error: {e}")

    # CLV prediction
    try:
        clv_params = params['tasks']['clv_prediction']
        clv_features_list = clv_params['numerical_features'] + clv_params['categorical_features']
        
        clv_input_data = input_data[clv_features_list]

        clv_features_processed = models["clv_preprocessor"].transform(clv_input_data)
        clv_pred = float(models["clv_model"].predict(clv_features_processed)[0])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CLV prediction error: {e}")

    # response
    return AnalysisResponse(
        segment_id=segment_id,
        segment_name=segment_name,        
        churn_probability=churn_prob,
        churn_decision=churn_decision,
        churn_risk_level=churn_risk_level,
        predicted_clv=clv_pred
    )