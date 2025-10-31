# src/pipeline/recommend.py

import pandas as pd
from pathlib import Path
import joblib
import argparse
from src.utils.logger import logger
from src.utils.common import read_yaml

# # forget this because recommendation and pricing is removed
class RecommendationPipeline:
    def __init__(self):
        self.config = read_yaml(Path("configs/config.yaml"))
        self.params = read_yaml(Path("configs/params.yaml"))
        
        model_path = Path(self.config['model_trainer']['trained_models_dir']) / "recommendation_system_model.pkl"
        self.model = joblib.load(model_path)
        
        order_data_path = Path(self.config['data_ingestion']['root_dir']) / self.params['data_ingestion']['source_order_filename']
        self.df = pd.read_csv(order_data_path)
        
        self.all_product_ids = self.df['product_id'].unique()
        
        logger.info("Recommendation Pipeline initialized successfully.")

    def generate_recommendations(self, customer_id: str, top_n: int = 10):
        """
        Generates a list of top_n product recommendations for a given customer.
        """
        try:
            products_bought_by_customer = self.df[self.df['customer_id'] == customer_id]['product_id'].unique()
            
            products_to_predict = [pid for pid in self.all_product_ids if pid not in products_bought_by_customer]
            
            logger.info(f"Generating recommendations for customer {customer_id} from {len(products_to_predict)} candidate products.")
            
            predictions = []
            for product_id in products_to_predict:
                pred = self.model.predict(uid=customer_id, iid=product_id)
                predictions.append((product_id, pred.est))
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            top_n_recommendations = [product_id for product_id, rating in predictions[:top_n]]
            
            return top_n_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for customer {customer_id}: {e}")
            raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate product recommendations for a customer.")
    parser.add_argument("--customer_id", type=str, required=True, help="The ID of the customer to generate recommendations for.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations to generate.")
    
    args = parser.parse_args()
    
    pipeline = RecommendationPipeline()
    recommendations = pipeline.generate_recommendations(args.customer_id, args.top_n)
    
    print(f"\n--- Top {args.top_n} Recommendations for Customer: {args.customer_id} ---")
    for i, product_id in enumerate(recommendations, 1):
        print(f"{i}. Product ID: {product_id}")