import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
bundle = joblib.load(os.path.join(BASE_DIR, "revenue_model.pkl"))

model = bundle["model"]
features = bundle["features"]

st.title("Amazon Revenue Predictor")

price = st.number_input("Price", min_value=0.0, value=200.0)
quantity = st.number_input("Quantity Sold", min_value=0.0, value=1.0)
discount = st.number_input("Discount %", min_value=0.0, max_value=100.0, value=10.0)
rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=4.0)
reviews = st.number_input("Review Count", min_value=0.0, value=100.0)

X_input = pd.DataFrame([{
    "price": price,
    "quantity_sold": quantity,
    "discount_percent": discount,
    "rating": rating,
    "review_count": reviews
}])[features]

if st.button("Predict Revenue"):
    pred = model.predict(X_input)[0]
    pred = max(0, float(pred))
    st.success(f"Predicted Total Revenue: {pred:,.2f}")
