import pandas as pd
import numpy as np
# import argparse # Ab iski zaroorat nahi
import json
from pathlib import Path
from src.pipeline.predict import PredictionPipeline # Prediction pipeline ko use karenge
from src.utils.logger import logger
import matplotlib.pyplot as plt


# forget this because recommendation and pricing is removed
# more precisely, full pipeline folder is removed but keep the code for future reference
def find_optimal_price(initial_data: dict, price_range_percent: float = 0.30, steps: int = 21):
    
    try:
        task_name = "conversion_prediction"
        logger.info(f"--- Product ke liye price optimization shuru ---")
        
        pipeline = PredictionPipeline(task_name=task_name)
    
        initial_price = initial_data[0]['price'] 
        min_price = initial_price * (1 - price_range_percent)
        max_price = initial_price * (1 + price_range_percent)
        price_points = np.linspace(min_price, max_price, steps)
        
        logger.info(f"Price test ho rahe hain: ${min_price:.2f} se ${max_price:.2f} tak ({steps} steps).")
        
        results = []
        
       
        for price in price_points:# 
            current_data = initial_data.copy() 
            current_data[0]['price'] = price
            
            result = pipeline.predict(current_data)
            probability = result['positive_class_probability'][0]
            expected_revenue = price * probability
            
            results.append({
                "price": price,
                "conversion_probability": probability,
                "expected_revenue": expected_revenue
            })
        results_df = pd.DataFrame(results)
        optimal_row = results_df.loc[results_df['expected_revenue'].idxmax()]
        
        logger.info(f"Optimization poora hua.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['price'], results_df['expected_revenue'], marker='o', linestyle='-')
        plt.title('Price vs. Expected Revenue')
        plt.xlabel('Price ($)')
        plt.ylabel('Expected Revenue ($)')
        plt.grid(True)
        plt.axvline(x=optimal_row['price'], color='r', linestyle='--', label=f"Optimal Price: ${optimal_row['price']:.2f}")
        plt.legend()
        plot_path = "optimal_price_curve.png"
        plt.savefig(plot_path)
        logger.info(f"Optimization curve plot save hua: {plot_path}")

        return optimal_row

    except Exception as e:
        logger.error(f"Price optimization mein error: {e}")
        raise e

if __name__ == '__main__':
    # tried everything then this is best for testing
    input_data_for_test = [
    {
        "price": 45.0,
        "freight_value": 8.0,
        "purchase_to_approval": 0.1,
        "approval_to_carrier": 0.8,
        "estimated_vs_actual": -3.0,
        "product_name_length": 50.0,
        "product_description_lenght": 400.0,
        "product_photos_qty": 2.0,
        "product_weight_g": 300.0,
        "product_volume_cm3": 1200.0,
        "price_X_sentiment": 45.0,  
        "price_X_photos_qty": 90.0,
        "payment_type": "voucher",
        "product_category_name": "books_technical",
        "seller_state": "PR",
        "region": "south",
        "review_sentiment": "positive"
    }
    ]

    try:
        optimal_price_details = find_optimal_price(input_data_for_test)
        
        print("\n--- Optimal Pricing Strategy ---")
        if input_data_for_test:
             print(f"Initial Price: ${input_data_for_test[0].get('price', 'N/A'):.2f}")
        else:
            print("Initial Price: N/A")
            
        print(f"Optimal Price: ${optimal_price_details['price']:.2f}")
        print(f"Conversion Probability at Optimal Price: {optimal_price_details['conversion_probability']:.2%}")
        print(f"Maximum Expected Revenue: ${optimal_price_details['expected_revenue']:.2f}")
        print("\nCheck 'optimal_price_curve.png' for the visualization.")

    except Exception as e:
         logger.error(f"Script run karne mein error: {e}")
         print(f"An error occurred: {e}")