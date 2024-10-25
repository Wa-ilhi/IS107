import streamlit as st
from streamlit_option_menu import option_menu
import dashboard  # Import your dashboard file
import datamining  # Import your datamining file

# Streamlit app title and configuration
st.set_page_config(page_title="Retail Analytics Application", layout="wide")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Data Visuals", "Data Mining Insights"],
        icons=["bar-chart", "search"],
        menu_icon="cast",
        default_index=0,
    )

# Load the selected page
if selected == "Data Visuals":
    st.title("Data Visuals")
    dashboard.main()  # Call the main function from dashboard.py

elif selected == "Data Mining Insights":
    st.title("Data Mining Insights")
    datamining.main()  # Call the main function from datamining.py
