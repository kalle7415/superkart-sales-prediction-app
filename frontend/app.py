import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend (the two containers talk to each other over
# the shared Docker network using the backend container's name)
BACKEND_URL = "http://backend:7860"

st.title("SuperKart Sales Forecasting")

# ---------------- Online Prediction ----------------
st.subheader("Online Prediction")

Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3f")
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=140.0)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
Product_Id_char = st.selectbox("Product Id Prefix", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age (years)", min_value=0, max_value=60, value=16, step=1)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

input_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category,
}

if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data)
    if response.status_code == 200:
        prediction = response.json()["Predicted_Sales"]
        st.success(f"Predicted Sales: {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# ---------------- Batch Prediction ----------------
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)
        else:
            st.error("Unable to connect to the prediction API.")
