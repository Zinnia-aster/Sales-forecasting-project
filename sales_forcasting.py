
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("🛒 Store Sales Forecasting App")
st.write("Predict store sales and visualize forecasts for the next month using CatBoost.")


# Load Dataset

uploaded_file = st.file_uploader("Upload your sales dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Check for sales column
    if 'sales' not in df.columns:
        st.error("❌ CSV must contain a 'Sales' column.")
        st.stop()
    
    sales = df['sales'].astype(float).values
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    
    # Create Lag Features
   
    n_lags = st.slider("Number of past sales to use for prediction", min_value=1, max_value=30, value=5)
    X, y = [], []
    for i in range(n_lags, len(sales)):
        X.append(sales[i-n_lags:i])
        y.append(sales[i])
    X, y = np.array(X), np.array(y)

    
    #  Train CatBoost Model
   
    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        loss_function='RMSE',
        verbose=100
    )
    model.fit(X, y)

   
    # Predict Future Sales
   
    st.subheader("🎯 Future Sales Forecast")
    n_future = st.slider("Number of future periods to predict", min_value=1, max_value=30, value=7)

    last_window = sales[-n_lags:].tolist()
    predictions = []
    for _ in range(n_future):
        pred = model.predict([last_window])[0]
        predictions.append(pred)
        last_window = last_window[1:] + [pred]

    forecast_df = pd.DataFrame({
        "Period": range(1, n_future+1),
        "Predicted Sales": predictions
    })
    st.dataframe(forecast_df)

    
    #  Visualize Forecast
    
    st.subheader("📈 Historical vs Forecasted Sales")
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(sales)+1), sales, label="Historical Sales", marker='o')
    plt.plot(range(len(sales)+1, len(sales)+n_future+1), predictions, label="Forecasted Sales", marker='x', linestyle='--')
    plt.xlabel("Period Index")
    plt.ylabel("Sales")
    plt.legend()
    st.pyplot(plt)


