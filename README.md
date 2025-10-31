# End-to-End Customer Analytics API (Churn, CLV & Segmentation)

This project documents the end-to-end journey of building a real-world customer analytics platform. It transforms raw, messy e-commerce data into a "smart" predictive API using FastAPI and Pydantic.

The final application analyzes customer data to predict **Churn Probability**, **Customer Lifetime Value (CLV)**, and **Customer Segment**, achieving a **93.3% ROC AUC** on churn prediction and a **0.916 R²** on CLV prediction.

---

## 📈 The Journey: A Real-World Data Challenge

This project was a deep dive into the frustrations and triumphs of real-world data science.

My initial goal was ambitious: to build **5 predictive models** (Churn, CLV, Segmentation, Dynamic Pricing, and Recommendation) from a single, complex dataset.

### The 4-Day Data Cleaning Marathon
The first challenge was data engineering. It took **4 days of intensive cleaning and merging in Google Colab** to wrangle 6 different raw CSVs (orders, items, payments, reviews, customers, etc.) into a single, usable `customer_ready.csv` file.

### The "Champion Model" Tuning
For the core models, I didn't just pick one algorithm. I ran a "bake-off" by tuning **Logistic Regression**, **RandomForest**, and **XGBoost** to find the absolute best performer for each task. We used SMOTE to solve class imbalance in churn and advanced feature engineering (like `tenure_X_spending`) to improve CLV.

* **Final Churn Model:** XGBoost (with SMOTE)
* **Final CLV Model:** XGBoost
* **Final Segmentation Model:** K-Means

### The Pivot: Why I Dropped 2 Models
This is where the project felt most like the real world. After successfully building 3 models, I had to drop Dynamic Pricing and Recommendation.

* **Dynamic Pricing Failure:** The `optimize_price` script consistently showed a linear relationship (always suggesting the highest price). This wasn't a model failure; it was a **data failure**. The dataset lacked the necessary price variation (i.e., the same product wasn't sold at different prices) for the model to learn price elasticity.
* **Recommendation Failure:** The user-item matrix was massive (`~98k users x 33k items`). This led to persistent `MemoryError`s. Specialized libraries like `scikit-surprise` and `implicit` also failed due to package installation and compilation errors (the infamous Cython/C++ build errors on Windows).

This was a critical lesson: **a data scientist must pivot when the data doesn't support the goal.** I'm proud of the 3 robust models we built, as they are based on data we can trust.

---

## 🛠️ Tech Stack & Architecture

* **Backend API:** **FastAPI**
* **Data Validation & Feature Creation:** **Pydantic** (using `@computed_field` to create 7+ features on the fly)
* **Model Training:** Scikit-learn, XGBoost, K-Means, Imbalanced-learn (SMOTE)
* **Data Handling:** Pandas, NumPy
* **Core Tools:** Joblib, PyYAML, Uvicorn, Git

### API Architecture

The final API is "smart." Instead of forcing the user to enter 17+ features:
1.  **User Input:** The client sends **10 simple "base" features** (like `total_orders`, `total_spent`, `first_order_date`).
2.  **API Logic (Pydantic):** The `CustomerBaseFeatures` model automatically calculates **7+ "derived" features** (like `avg_price`, `purchase_month`, `tenure_X_spending`) in the background.
3.  **Prediction Pipeline:** The API then runs this complete 17-feature set through the models in sequence (Segmentation -> Churn -> CLV) and returns a single, clean JSON response.



---

## 📂 Project Structure


├── api/
│   └── main.py             # The "smart" FastAPI application
├── app/
│   └── app_ui.py           # The Streamlit user interface
├── artifacts/
│   ├── .gitkeep
    |- data_ingetion
    |- data_preperation
    |- data_preprocessing
│   └── models/             # Saved Champion Models (.pkl) and Preprocessors (.pkl)
├── configs/
│   ├── config.yaml         # File paths and directories
│   └── params.yaml         # All hyperparameters and feature lists
├── src/
│   ├── components/         # Data Ingestion & Preprocessing scripts
│   ├── pipeline/           # Old prediction scripts (optimize_price, etc.)
│   ├── training_scripts/   # Scripts to train the final "Champion" models
│   ├── tuning/             # just test file 
│   └── utils/              # Helper functions (logger, common utils)
├── .gitignore              # Ignores data, venv, artifacts, etc.
├── requirements.txt        # All Python libraries to install
└── run_pipeline.py         # Master script to run data ingestion & preprocessing


---
## 🚀 How to Run

### 1. Data Preparation (Crucial Step)

This repository **does not include the data files**.

The data pipeline depends on two cleaned CSVs that were created in Google Colab from 6 raw files. To run this project, you must provide your own cleaned CSVs in the `artifacts/data_prepration` folder:

1.  **`customer_ready.csv`:** A file where each row is one customer, containing all their historical features (e.g., `total_orders`, `total_spent`, `avg_review_score`, `customer_tenure_days`, `segment_id`, etc.).
2.  **`order_master_final.csv`:** A file where each row is an order item, containing all order-level details (e.g., `price`, `freight_value`, `product_category_name`, `review_sentiment`, etc.).

*The column names in your files must match the feature names defined in the `params.yaml` file for the scripts to work.*

### 2. Setup and Training

```bash
# 1. Clone the repo and create environment
git clone https://github.com/vansh-tyagi-ji/customer-analytics-API.git
cd YOUR_REPO_NAME
python -m venv venv_py311
.\venv_py311\Scripts\activate

# 2. Install requirements
pip install -r requirements.txt

# 3. (Important) Place your cleaned CSVs in the artifacts/data_preperation folder

# 4. Run the preprocessing pipeline
python run_pipeline.py

# 5. Train all the final "champion" models
python -m src.training_scripts.train_champion_segmentation
python -m src.training_scripts.train_champion_churn
python -m src.training_scripts.train_champion_clv

Run the Application
Run the backend and frontend in two separate terminals.

Terminal 1 (Backend API):
Bash
python -m uvicorn api.main:app --reload
# API is now live at http://127.0.0.1:8000

Terminal 2 (Frontend UI):
Bash
python -m streamlit run app/app_ui.py
# UI is now live at http://127.0.0.1:8501
