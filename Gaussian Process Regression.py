import streamlit as st
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
import yfinance as yf

def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close'].values

def create_features(data, lookback_period):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.reshape(-1, 1)
    return X, y

def perform_gpr(X, y, X_pred, length_scale, noise_variance):
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_variance)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(X, y)
    y_pred, sigma = gpr.predict(X_pred, return_std=True)
    return y_pred, sigma

def main():
    st.title("Gaussian Process Regression Indicator")

    # Sidebar inputs
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
    lookback_period = st.sidebar.slider("Lookback Period", 10, 200, 100)
    prediction_horizon = st.sidebar.slider("Prediction Horizon", 1, 50, 30)
    length_scale = st.sidebar.slider("Length Scale", 1.0, 20.0, 10.0)
    noise_variance = st.sidebar.slider("Noise Variance", 0.01, 1.0, 0.1, 0.01)

    # Load data
    data = load_data(symbol, start_date, end_date)

    # Create features
    X, y = create_features(data, lookback_period)

    # Prepare prediction range
    X_pred = np.arange(len(data) + prediction_horizon).reshape(-1, 1)

    # Perform GPR
    y_pred, sigma = perform_gpr(X, y, X_pred, length_scale, noise_variance)

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(X, y, label='Actual', color='blue')
    ax.plot(X_pred, y_pred, label='GPR mean', color='red')
    ax.fill_between(X_pred.ravel(), y_pred.ravel() - 1.96 * sigma, y_pred.ravel() + 1.96 * sigma,
                    alpha=0.2, color='red', label='95% confidence interval')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Generate alert
    if len(y_pred) > len(y):
        last_actual = y[-1][0]
        last_forecast = y_pred[-1][0]
        if last_forecast > last_actual:
            st.success("GPR Bullish Forecast")
        elif last_forecast < last_actual:
            st.error("GPR Bearish Forecast")
        else:
            st.info("GPR Neutral Forecast")

if __name__ == "__main__":
    main()