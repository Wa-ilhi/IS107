import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

def main():
    # st.title("Data Mining Insights")
    # st.write("Loading cleaned data...")
    data = pd.read_excel('Cleaned_Online_Retail.xlsx')

    # st.write("Data cleaning...")
    data.dropna(subset=['CustomerID', 'UnitPrice', 'Quantity'], inplace=True)
    data['TotalSales'] = data['Quantity'] * data['UnitPrice']

    # Customer Segmentation using K-Means Clustering
    st.subheader("Customer Segmentation using K-Means Clustering")
    customer_data = data.groupby('CustomerID').agg({
        'TotalSales': 'sum',
        'InvoiceNo': 'count'
    }).reset_index()

    customer_data.columns = ['CustomerID', 'TotalSales', 'PurchaseCount']
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data[['TotalSales', 'PurchaseCount']])

    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(customer_data['TotalSales'], customer_data['PurchaseCount'], c=customer_data['Cluster'], cmap='viridis')
    ax.set_xlabel('Total Sales')
    ax.set_ylabel('Purchase Count')
    ax.set_title('Customer Segmentation Using K-Means Clustering')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    # Predictive Analysis using Linear Regression
    st.subheader("Predictive Analysis using Linear Regression")
    X = data[['Quantity', 'UnitPrice']]
    y = data['TotalSales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Plot Actual vs. Predicted Total Sales
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel('Actual Total Sales')
    ax.set_ylabel('Predicted Total Sales')
    ax.set_title('Actual vs Predicted Total Sales')

    # Plot the ideal regression line (y = x)
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label='Ideal: y = x')

    # Add legend
    ax.legend()

    st.pyplot(fig)

if __name__ == '__main__':
    main()
