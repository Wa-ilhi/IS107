import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
import psycopg2

# Function to connect to PostgreSQL and fetch data from the cleaned_data table
def get_data_from_db():
    # Database connection parameters
    host = 'localhost'  # Use 'localhost' if PostgreSQL is running on your local machine
    database = 'db107'  # Your database name
    user = 'postgres'    # Default PostgreSQL user
    password = '123'     # Your password for the 'postgres' user
    
    # Establish connection to PostgreSQL
    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    
    # Query to fetch data from the cleaned_data table
    query = "SELECT * FROM cleaned_data;"
    
    # Load data into a pandas DataFrame
    data = pd.read_sql(query, conn)
    
    # Close the database connection
    conn.close()
    
    return data

def main():
    # Fetch data from PostgreSQL
    data = get_data_from_db()

    # Data cleaning
    data.dropna(subset=['customerid', 'unitprice', 'quantity'], inplace=True)  # Adjust column names to match database schema
    data['totalsales'] = data['quantity'] * data['unitprice']

    # Customer Segmentation using K-Means Clustering
    st.subheader("Customer Segmentation using K-Means Clustering")
    customer_data = data.groupby('customerid').agg({
        'totalsales': 'sum',
        'invoiceno': 'count'
    }).reset_index()

    customer_data.columns = ['CustomerID', 'TotalSales', 'PurchaseCount']
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data[['TotalSales', 'PurchaseCount']])

    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

    # Plot the K-Means clusters with color labels
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(customer_data['TotalSales'], customer_data['PurchaseCount'], c=customer_data['Cluster'], cmap='viridis')
    ax.set_xlabel('Total Sales')
    ax.set_ylabel('Purchase Count')
    ax.set_title('Customer Segmentation Using K-Means Clustering')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    st.markdown("""
    ### Cluster Legend:
    - **Violet**: Low spenders, few purchases.
    - **Yellow**: Moderate spenders, high purchase count.
    - **Green**: Low spenders, lower purchase count.
    - **Blue**: High spenders, scattered purchase count.
    """)

    # Predictive Analysis using Linear Regression
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Predictive Analysis using Linear Regression")
    X = data[['quantity', 'unitprice']]
    y = data['totalsales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Get the model coefficients and intercept
    intercept = model.intercept_
    coef_quantity = model.coef_[0]
    coef_unitprice = model.coef_[1]

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

    # Add the regression formula and MSE as annotations inside the plot
    formula_text = f"y = {intercept:.2f} + ({coef_quantity:.2f} * quantity) + ({coef_unitprice:.2f} * unitprice)"
    mse_text = f"MSE: {mse:.2f}"

    # Add the text to the plot (placing it at specific locations in the graph)
    ax.text(0.05, 0.95, formula_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black', fontweight='bold')
    ax.text(0.05, 0.90, mse_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black', fontweight='bold')

    # Add legend for the ideal line
    ax.legend()

    st.pyplot(fig)

    # Slope Interpretation placed below the graph
    slope_legend_text = f"### Slope Interpretation:\n"
    slope_legend_text += f"- If 'quantity' slope is positive: More items sold increase total sales.\n"
    slope_legend_text += f"- If 'unitprice' slope is positive: Higher price per unit increases total sales.\n"
    
    st.markdown(slope_legend_text)

if __name__ == '__main__':
    main()
