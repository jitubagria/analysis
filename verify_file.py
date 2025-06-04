import pandas as pd
import sys

print("Rechecking the Excel file...")

# Check original file first
try:
    df = pd.read_excel('attached_assets/overall_master.xlsx')
    print("\nOriginal file (overall_master.xlsx):")
    print(f"Number of rows: {len(df)}")
    print(f"System values: {df['System'].value_counts().to_dict()}")
except Exception as e:
    print(f"Error reading original file: {str(e)}")

# Check if the test file exists
try:
    test_df = pd.read_excel('attached_assets/all_systems_test.xlsx')
    print("\nTest file (all_systems_test.xlsx):")
    print(f"Number of rows: {len(test_df)}")
    print(f"System values: {test_df['System'].value_counts().to_dict()}")
except Exception as e:
    print(f"Error reading test file: {str(e)}")

# Check if an export file exists
try:
    export_df = pd.read_csv('attached_assets/2025-05-09T15-49_export.csv')
    print("\nExport file (2025-05-09T15-49_export.csv):")
    print(f"Number of rows: {len(export_df)}")
    if 'System' in export_df.columns:
        print(f"System values: {export_df['System'].value_counts().to_dict()}")
    else:
        print("System column not found in export file")
        print("Columns in export file:", export_df.columns.tolist())
except Exception as e:
    print(f"Error reading export file: {str(e)}")