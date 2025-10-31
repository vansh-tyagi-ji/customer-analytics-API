# app_ui.py

import streamlit as st
import requests
import json
from datetime import date

st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
st.title("👨‍💼 Customer Analytics Dashboard")
st.markdown("Enter customer details to get Segment, Churn, and CLV predictions.")

API_URL = "http://127.0.0.1:8000/analyze_customer/"


# st.sidebar.header("Input Customer Features")
with st.form(key="customer_form"):
    st.subheader("Enter Customer Details")
    st.info("For a **New Customer**, just leave the values as 0.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_orders = st.number_input("Total Orders", min_value=0, value=0)
        total_spent = st.number_input("Total Spent ($)", min_value=0.0, value=0.0, format="%.2f")
        total_items = st.number_input("Total Items Purchased", min_value=0, value=0)
        
    with col2:
        days_since_last_order = st.number_input("Days Since Last Order", min_value=0, value=0)
        customer_tenure_days = st.number_input("Customer Tenure (Days)", min_value=0, value=0)
        total_freight = st.number_input("Total Freight Cost ($)", min_value=0.0, value=0.0, format="%.2f")

    with col3:
        avg_review_score = st.slider("Average Review Score", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
        customer_state = st.text_input("Customer State (e.g., SP)", value="SP")
        review_speed_category = st.selectbox("Review Speed", ["Normal", "Quick", "Slow", "Instant"], index=0)
        first_order_date = st.date_input("First Order Date", value=date.today())

    submit_button = st.form_submit_button(label="Analyze Customer")

# 
if submit_button:
    # prepare data for API call
    with st.spinner("Analyzing customer data..."):
    #   input data preparation
     input_data = {
        "days_since_last_order": days_since_last_order,
        "total_orders": total_orders,
        "total_spent": total_spent,
        "avg_review_score": avg_review_score,
        "customer_tenure_days": customer_tenure_days,
        "total_items": total_items,
        "total_freight": total_freight,
        "customer_state": customer_state,
        "review_speed_category": review_speed_category,
        "first_order_date": first_order_date.isoformat()
    }

    #  api call
    try:
        response = requests.post(API_URL, data=json.dumps(input_data))
        response.raise_for_status() 
        
        results = response.json()

        # Display Results
        st.subheader("📈 Customer Analysis Results")
        col1, col2, col3 = st.columns(3)

        # Metric 1: Customer Segment
        col1.metric(label="Customer Segment", value=results['segment_name'])
        
        
        churn_prob_percent = results['churn_probability'] * 100
        churn_risk = results['churn_risk_level']
        delta_color = "inverse" if churn_risk == "High Risk" else "normal"
        
        col2.metric(
            label="Churn Risk", 
            value=f"{churn_prob_percent:.1f}%", 
            delta=churn_risk,
            delta_color=delta_color
        )
        
        clv_value = results['predicted_clv']
        col3.metric(label="Predicted CLV", value=f"${clv_value:.2f}")


        # tokk some help from chatgpt for error lines messages
        st.subheader("💡 Next Best Action")
        if results['churn_decision'] == "Churn":
            st.error(f"**Action:** This customer is a **High Churn Risk**! Recommend sending a retention offer or discount coupon.")
        elif results['segment_name'] == "Champions":
            st.success(f"**Action:** This is a **Champion Customer**! Recommend offering a loyalty reward or early access to new products.")
        elif results['segment_name'] == "At-Risk Customers":
             st.warning(f"**Action:** This customer is **At-Risk**. Recommend starting a re-engagement email campaign.")
        else:
            st.info(f"**Action:** This is a **Loyal/New Customer**. Recommend standard marketing or 'you might also like' cross-sells.")

    except requests.exceptions.HTTPError as err:
        st.error(f"API Error: {err.response.status_code} - {err.response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: Could not connect to the API at {API_URL}. Kya aapne `uvicorn main:app --reload` chalaya hai?")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")