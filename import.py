import pandas as pd

# Load the cleaned Excel file
data = pd.read_excel('Cleaned_Online_Retail.xlsx')

# Export to CSV
data.to_csv('cleaned_data.csv', index=False)
