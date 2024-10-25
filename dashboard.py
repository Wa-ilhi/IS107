import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def main():
    # Load data
    data = pd.read_excel('Cleaned_Online_Retail.xlsx')

    # Data cleaning
    data.dropna(subset=['CustomerID'], inplace=True)
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    # Calculate total sales
    data['TotalSales'] = data['Quantity'] * data['UnitPrice']
    data['TotalSales'] = pd.to_numeric(data['TotalSales'], errors='coerce')

    # Total sales metric
    total_sales = data['TotalSales'].sum()

    # Display total sales
    st.metric("Total Sales", f"£{total_sales:.2f}")

    # Top-selling products
    top_products = data.groupby('Description')['TotalSales'].sum().nlargest(10).sort_values(ascending=False)

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
    start_date = st.date_input("Start date", data['InvoiceDate'].min())
    end_date = st.date_input("End date", data['InvoiceDate'].max())

    # Filter data based on date range
    filtered_data = data[(data['InvoiceDate'] >= pd.to_datetime(start_date)) & 
                         (data['InvoiceDate'] <= pd.to_datetime(end_date))]

    # Update total sales based on filtered data
    total_sales_filtered = filtered_data['TotalSales'].sum()
    st.metric("Total Sales (Filtered)", f"£{total_sales_filtered:.2f}")

    # Show filtered top products
    filtered_top_products = filtered_data.groupby('Description')['TotalSales'].sum().nlargest(10).sort_values(ascending=False)
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
