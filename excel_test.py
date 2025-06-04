import pandas as pd
import numpy as np
import streamlit as st

# Read the Excel file
file_path = 'attached_assets/overall_master.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Excel file read successfully")
    print("Columns in the Excel file:", df.columns.tolist())
    print(f"Number of rows in file: {len(df)}")
    
    # Check for a System column
    if 'System' in df.columns:
        system_values = df['System'].value_counts().to_dict()
        print(f"System column found with values and counts: {system_values}")
        
        # Check if there are different system types
        if len(system_values) < 3:
            print("\nWARNING: You mentioned there should be 3 types (amicus, comtect, optia)")
            print("But only found:", system_values.keys())
            # Let's check if the data is filtered
            print("\nIs this data already filtered to just one system type?")
        
        # Count rows by System value
        print("\nBreakdown by System type:")
        for system, count in system_values.items():
            print(f"  {system}: {count} rows")
        
        # Show some sample rows for each type
        unique_systems = df['System'].unique()
        for system in unique_systems:
            system_df = df[df['System'] == system]
            print(f"\nSample rows for System = {system}:")
            print(system_df.head(2))
    else:
        print("System column NOT FOUND - checking other column names...")
        for col in df.columns:
            print(f"Column: {col}")
            if 'system' in col.lower():
                print(f"  Possible system column with values: {df[col].unique().tolist()}")
    
except Exception as e:
    print(f"Error reading Excel file: {str(e)}")