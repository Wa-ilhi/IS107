import pandas as pd
import psycopg2
import streamlit as st
import matplotlib.pyplot as plt

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
    # Load data from PostgreSQL
    data = get_data_from_db()

    # Data cleaning
    data.dropna(subset=['customerid'], inplace=True)  # Use 'customerid' (lowercase) here
    data['invoicedate'] = pd.to_datetime(data['invoicedate'])  # Use 'invoicedate' (lowercase)

    # Calculate total sales
    data['totalsales'] = data['quantity'] * data['unitprice']  # Use 'quantity' and 'unitprice' (lowercase)
    data['totalsales'] = pd.to_numeric(data['totalsales'], errors='coerce')

    # Total sales metric
    total_sales = data['totalsales'].sum()

    # Display total sales
    st.metric("Total Sales", f"£{total_sales:.2f}")

    # Top-selling products
    top_products = data.groupby('description')['totalsales'].sum().nlargest(10).sort_values(ascending=False)  # Use 'description' (lowercase)

    # Visualize top-selling products
    st.subheader("Top Selling Products")
    plt.figure(figsize=(10, 6))
    top_products.plot(kind='bar', color='skyblue')
    plt.title('Top Selling Products')
    plt.xlabel('Products')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

    # Date filter
    start_date = st.date_input("Start date", data['invoicedate'].min())  # Use 'invoicedate' (lowercase)
    end_date = st.date_input("End date", data['invoicedate'].max())  # Use 'invoicedate' (lowercase)

    # Filter data based on date range
    filtered_data = data[(data['invoicedate'] >= pd.to_datetime(start_date)) & 
                         (data['invoicedate'] <= pd.to_datetime(end_date))]

    # Update total sales based on filtered data
    total_sales_filtered = filtered_data['totalsales'].sum()
    st.metric("Total Sales (Filtered)", f"£{total_sales_filtered:.2f}")

    # Show filtered top products
    filtered_top_products = filtered_data.groupby('description')['totalsales'].sum().nlargest(10).sort_values(ascending=False)  # Use 'description' (lowercase)
    st.subheader("Top Selling Products (Filtered)")
    plt.figure(figsize=(10, 6))
    filtered_top_products.plot(kind='bar', color='salmon')
    plt.title('Top Selling Products (Filtered)')
    plt.xlabel('Products')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

if __name__ == '__main__':
    main()



#streamlit run dashboard.py to run the dashboard --Terminal/not powershell--streamlit run dashboard.py---- visualization-----
