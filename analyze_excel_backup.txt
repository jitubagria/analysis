import pandas as pd
import numpy as np
from openpyxl import load_workbook
import os

def analyze_excel_file(file_path):
    """
    Analyze all sheets in an Excel file and report on inputs, processing, and outputs.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
        
    Returns:
    --------
    dict
        Dictionary containing analysis of all sheets
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return {"error": f"File {file_path} not found"}
    
    try:
        # Load all sheets with pandas
        excel_data = pd.ExcelFile(file_path)
        sheet_names = excel_data.sheet_names
        
        # Use openpyxl to get more metadata
        wb = load_workbook(file_path, read_only=True, data_only=True)
        
        results = {
            "file_name": os.path.basename(file_path),
            "total_sheets": len(sheet_names),
            "sheet_names": sheet_names,
            "sheets": {}
        }
        
        # Analyze each sheet
        for sheet_name in sheet_names:
            print(f"Analyzing sheet: {sheet_name}")
            
            # Read sheet data
            try:
                df = pd.read_excel(excel_data, sheet_name=sheet_name)
                
                # Basic sheet info
                sheet_info = {
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "column_names": df.columns.tolist(),
                    "num_numeric_cols": len(df.select_dtypes(include=['number']).columns),
                    "num_text_cols": len(df.select_dtypes(include=['object']).columns),
                    "missing_values": df.isna().sum().sum(),
                    "categorical_columns": [],
                    "numerical_columns": [],
                    "classification": "Unknown"
                }
                
                # Classify columns
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        col_info = {
                            "name": col,
                            "dtype": str(df[col].dtype),
                            "missing": df[col].isna().sum(),
                            "unique_values": df[col].nunique(),
                            "min": float(df[col].min()) if not df[col].empty and not pd.isna(df[col].min()) else None,
                            "max": float(df[col].max()) if not df[col].empty and not pd.isna(df[col].max()) else None,
                            "mean": float(df[col].mean()) if not df[col].empty and not pd.isna(df[col].mean()) else None
                        }
                        sheet_info["numerical_columns"].append(col_info)
                    else:
                        # For categorical columns, get top values
                        value_counts = df[col].value_counts().head(5).to_dict()
                        col_info = {
                            "name": col,
                            "dtype": str(df[col].dtype),
                            "missing": int(df[col].isna().sum()),
                            "unique_values": int(df[col].nunique()),
                            "top_values": {str(k): int(v) for k, v in value_counts.items()}
                        }
                        sheet_info["categorical_columns"].append(col_info)
                
                # Try to classify sheet purpose based on column names and patterns
                sheet_purpose = classify_sheet_purpose(df, sheet_name)
                sheet_info["classification"] = sheet_purpose
                
                # Identify potential input areas
                sheet_info["inputs"] = identify_inputs(df)
                
                # Identify potential calculated fields (outputs)
                sheet_info["outputs"] = identify_outputs(df)
                
                # Identify potential processing or calculations
                sheet_info["processing"] = identify_processing(df)
                
                results["sheets"][sheet_name] = sheet_info
            
            except Exception as e:
                results["sheets"][sheet_name] = {
                    "error": str(e)
                }
        
        return results
    
    except Exception as e:
        return {"error": str(e)}

def classify_sheet_purpose(df, sheet_name):
    """Attempt to classify the purpose of a sheet based on its content"""
    
    # Check sheet name for clues
    name_lower = sheet_name.lower()
    
    if any(keyword in name_lower for keyword in ['raw', 'data', 'input']):
        return "Data Input"
    
    if any(keyword in name_lower for keyword in ['config', 'setting', 'parameter']):
        return "Configuration"
    
    if any(keyword in name_lower for keyword in ['output', 'result', 'summary', 'report']):
        return "Results/Output"
    
    if any(keyword in name_lower for keyword in ['calc', 'formula', 'process']):
        return "Calculation/Processing"
    
    # Check column names for clues
    col_names = ' '.join(df.columns.astype(str)).lower()
    
    if any(keyword in col_names for keyword in ['id', 'key', 'name', 'date', 'time']):
        # Common in raw data tables
        if 'result' not in col_names and 'output' not in col_names:
            return "Data Input"
    
    if any(keyword in col_names for keyword in ['result', 'output', 'score', 'metric']):
        return "Results/Output"
    
    if any(keyword in col_names for keyword in ['parameter', 'setting', 'config']):
        return "Configuration"
    
    # Check for presence of formulas (not always possible in read_only mode)
    
    # Default classification
    return "General Purpose"

def identify_inputs(df):
    """Identify potential input fields in the dataframe"""
    inputs = []
    
    # Look for columns that appear to be inputs
    for col in df.columns:
        # Convert column name to string for safer comparison
        col_str = str(col).lower()
        
        # Check column name for clues
        if any(keyword in col_str for keyword in ['input', 'parameter', 'setting', 'id', 'name', 'date']):
            inputs.append({
                "name": str(col),
                "reason": "Column name suggests input data"
            })
        
        # Check for categorical variables with few unique values
        # These are often selection inputs
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < 10:
            inputs.append({
                "name": str(col),
                "reason": "Categorical variable with limited options"
            })
    
    return inputs

def identify_outputs(df):
    """Identify potential output fields in the dataframe"""
    outputs = []
    
    # Look for columns that appear to be outputs
    for col in df.columns:
        # Convert column name to string for safer comparison
        col_str = str(col).lower()
        
        # Check column name for clues
        if any(keyword in col_str for keyword in ['output', 'result', 'score', 'calculated', 'total', 'sum']):
            outputs.append({
                "name": str(col),
                "reason": "Column name suggests calculated output"
            })
        
        # Check for columns with % or ratio in name
        if any(keyword in col_str for keyword in ['%', 'percent', 'ratio', 'rate']):
            outputs.append({
                "name": str(col),
                "reason": "Appears to be a calculated percentage or ratio"
            })
    
    return outputs

def identify_processing(df):
    """Identify potential processing or calculation elements"""
    processing = []
    
    # Look for numeric columns that appear to be calculations
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Look for patterns in numeric columns
    for col in numeric_cols:
        col_str = str(col).lower()
        
        # Columns with math operations in their names
        if any(op in col_str for op in ['sum', 'avg', 'average', 'calc', 'mean', 'total', 'diff', 'delta']):
            processing.append({
                "name": str(col),
                "operation": "Mathematical calculation",
                "reason": "Column name indicates calculated value"
            })
        
        # Columns with % - likely percentage calculations
        if '%' in col_str or 'percent' in col_str:
            processing.append({
                "name": str(col),
                "operation": "Percentage calculation",
                "reason": "Column name indicates percentage"
            })
        
        # Check for pre/post pairs - common in analyses
        if 'pre' in col_str:
            post_version = col_str.replace('pre', 'post')
            if any(post_version == str(other_col).lower() for other_col in df.columns):
                processing.append({
                    "name": f"{str(col)} -> {post_version}",
                    "operation": "Pre/Post comparison",
                    "reason": "Matched pre/post column pair"
                })
    
    return processing

if __name__ == "__main__":
    # Analyze both Excel files
    files = [
        "attached_assets/overall_master.xlsx",
        "attached_assets/all_systems_test.xlsx"
    ]
    
    for file_path in files:
        results = analyze_excel_file(file_path)
        
        # Print a summary of findings
        print(f"\n=== Excel File Analysis Results: {os.path.basename(file_path)} ===")
        print(f"File: {results['file_name']}")
        print(f"Number of sheets: {results['total_sheets']}")
        
        print("\nSheet Summary:")
        for sheet_name, sheet_data in results['sheets'].items():
            if 'error' in sheet_data:
                print(f"  - {sheet_name}: ERROR - {sheet_data['error']}")
            else:
                classification = sheet_data.get('classification', 'Unknown')
                rows = sheet_data.get('rows', 0)
                cols = sheet_data.get('columns', 0)
                print(f"  - {sheet_name}: {classification} ({rows} rows, {cols} columns)")
                
                # Print inputs
                if sheet_data.get('inputs'):
                    print(f"    Inputs: {', '.join([i['name'] for i in sheet_data['inputs'][:5]])}")
                    if len(sheet_data['inputs']) > 5:
                        print(f"    ...and {len(sheet_data['inputs']) - 5} more")
                
                # Print outputs  
                if sheet_data.get('outputs'):
                    print(f"    Outputs: {', '.join([o['name'] for o in sheet_data['outputs'][:5]])}")
                    if len(sheet_data['outputs']) > 5:
                        print(f"    ...and {len(sheet_data['outputs']) - 5} more")
                
                # Numerical column stats
                num_cols = len(sheet_data.get('numerical_columns', []))
                cat_cols = len(sheet_data.get('categorical_columns', []))
                if num_cols > 0:
                    print(f"    Data: {num_cols} numerical columns, {cat_cols} categorical columns")
                
                # Print processing
                if sheet_data.get('processing'):
                    proc_list = sheet_data['processing']
                    print(f"    Processing: {len(proc_list)} calculation patterns including:")
                    for proc in proc_list[:3]:
                        print(f"      - {proc['name']}: {proc['operation']}")
                    if len(proc_list) > 3:
                        print(f"      - ...and {len(proc_list) - 3} more calculations")