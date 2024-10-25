import pandas as pd

# Step 1: Extract - Load data from the Excel file
file_path = 'Online-Retail.xlsx'
data = pd.read_excel(file_path)

# Step 2: Transform - Clean and prepare the data

# 2.1 Handle missing values
# Drop rows with missing CustomerID
data = data.dropna(subset=['CustomerID'])

# Fill missing descriptions
data['Description'] = data['Description'].fillna('No Description')

# Remove rows with negative quantities
data = data[data['Quantity'] > 0]

# 2.2 Convert 'InvoiceDate' to 'YYYY-MM-DD' format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate']).dt.strftime('%Y-%m-%d')

# 2.3 Remove outliers using the 1st and 99th percentiles
# Calculate the thresholds
quantity_thresholds = data['Quantity'].quantile([0.01, 0.99])
unitprice_thresholds = data['UnitPrice'].quantile([0.01, 0.99])

# Filter rows to keep only values within the defined percentile range
data = data[
    (data['Quantity'].between(quantity_thresholds.iloc[0], quantity_thresholds.iloc[1])) &
    (data['UnitPrice'].between(unitprice_thresholds.iloc[0], unitprice_thresholds.iloc[1]))
]

# 2.4 Add 'TotalPrice' column
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# Step 3: Load - Save the cleaned data to a new Excel file
output_file_path = 'Cleaned_Online_Retail.xlsx'
data.to_excel(output_file_path, index=False)
