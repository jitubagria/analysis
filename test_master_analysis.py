import streamlit as st
import pandas as pd
import numpy as np
from data_utils import analyze_master_parameters, get_descriptive_stats

st.title("Master Parameter Analysis Test")

# Load the test data
try:
    # Try all systems test file first
    df = pd.read_excel('attached_assets/all_systems_test.xlsx')
    st.success("Using test file with all three systems!")
except Exception:
    # If that fails, use the original file
    df = pd.read_excel('attached_assets/overall_master.xlsx')
    st.warning("Using original Excel file with only Amicus system")

# Display basic info
st.write(f"Data shape: {df.shape}")
st.write("System values:", df['System'].value_counts().to_dict())

# Display a sample of the data
st.subheader("Data Sample")
st.dataframe(df.head())

# Test master parameter analysis
st.subheader("Testing Master Parameter Analysis")

# Use System column as master column
master_columns = ['System']
st.write("Using master columns:", master_columns)

# Run the analysis
with st.spinner("Running analysis..."):
    master_results = analyze_master_parameters(df, master_columns)

# Show the results
if master_results:
    st.success("Analysis completed successfully!")
    st.write("Master results keys:", list(master_results.keys()))
    
    # Display analysis for each master column
    for master_col, results in master_results.items():
        if results.get('type') == 'error':
            st.error(f"Error analyzing {master_col}: {results.get('error')}")
            continue
            
        st.write(f"### Analysis by {master_col}")
        col_type = results.get('type', 'unknown')
        st.write(f"Column type: {col_type}")
        
        # Display results for each group
        for group_name, group_data in results.get('groups', {}).items():
            with st.expander(f"{master_col} = {group_name} (n={group_data['count']})"):
                group_stats = group_data.get('stats')
                if group_stats is not None:
                    # Convert to string for display
                    for col in group_stats.columns:
                        if 'Mean \u00B1 Std Dev' in group_stats.index and isinstance(group_stats.loc['Mean \u00B1 Std Dev', col], str):
                            group_stats[col] = group_stats[col].astype(str)
                    
                    st.dataframe(group_stats)
                else:
                    st.write("No statistics available for this group.")
else:
    st.error("Analysis failed to return results")