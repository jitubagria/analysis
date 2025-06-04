import pandas as pd
import numpy as np
import io
import re

def load_data(uploaded_file):
    """
    Load data from uploaded file (CSV, Excel)
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The file uploaded by the user
        
    Returns:
    --------
    pd.DataFrame, str
        The loaded dataframe and the filename
    """
    filename = uploaded_file.name
    file_ext = filename.split('.')[-1].lower()
    
    if file_ext == 'csv':
        try:
            # First try with pandas' automatic delimiter detection
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except UnicodeDecodeError as e:
            # Try with different encodings and delimiter detection
            encodings_to_try = ['latin1', 'cp1252', 'iso-8859-1', 'utf-16']
            separators_to_try = [None, ',', ';', '\t', ' ', '|']  # None means auto-detect
            df = None
            
            for encoding in encodings_to_try:
                for sep in separators_to_try:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        if sep is None:
                            # Use automatic delimiter detection
                            df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
                        else:
                            df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
                        
                        # Check if we got multiple columns (successful parsing)
                        if df.shape[1] > 1:
                            break
                    except:
                        continue
                if df is not None and df.shape[1] > 1:
                    break
            
            if df is None or df.shape[1] == 1:
                raise Exception(f"Failed to load CSV with proper column separation. Encoding error: {str(e)}")
        except Exception as e:
            # Final fallback: try different separators with default encoding
            separators_to_try = [',', ';', '\t', ' ', '|']
            df = None
            
            for sep in separators_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=sep)
                    if df.shape[1] > 1:  # Successfully separated into multiple columns
                        break
                except:
                    continue
                    
            if df is None:
                raise Exception(f"Failed to load CSV: {str(e)}")
    
    elif file_ext in ['xlsx', 'xls']:
        try:
            # Read excel with options to preserve original number format
            df = pd.read_excel(
                uploaded_file, 
                engine='openpyxl',  # Use openpyxl engine for better format control
                keep_default_na=True,
                na_values=['NA']  # Only treat 'NA' as missing, preserve zeros
            )
            
            # Fix possible mixed data types in columns by converting them to strings
            # This solves issues with columns containing both numbers and strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Convert all mixed object columns to strings to avoid PyArrow errors
                    df[col] = df[col].astype(str)
            
        except Exception as e:
            raise Exception(f"Failed to load Excel file: {str(e)}")
    
    else:
        raise Exception(f"Unsupported file format: {file_ext}")
    
    # Basic data cleaning
    # Replace empty strings with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Convert column names to string and fix encoding issues
    df.columns = df.columns.astype(str)
    
    # Fix common encoding issues in column names
    new_columns = []
    for col in df.columns:
        # Fix common encoding issues step by step
        fixed_col = str(col)
        
        # First, handle problematic encoding patterns
        fixed_col = fixed_col.replace('Ã10Â³/Â', '×10³/µL')
        fixed_col = fixed_col.replace('Ã10Â³/', '×10³/µL')
        
        # General encoding fixes
        fixed_col = fixed_col.replace('Ã', '×')
        fixed_col = fixed_col.replace('Â³', '³')
        fixed_col = fixed_col.replace('Â²', '²')
        fixed_col = fixed_col.replace('Â', '')
        
        # Handle incomplete unit patterns - but avoid double µL
        if '×10³/' in fixed_col and not fixed_col.endswith('µL') and 'µL' not in fixed_col:
            fixed_col = fixed_col.replace('×10³/', '×10³/µL')
        
        # Fix double µL issue and parentheses problems
        fixed_col = fixed_col.replace('µLµL', 'µL')
        fixed_col = fixed_col.replace('(×10³/µLµL)', '(×10³/µL)')
        fixed_col = fixed_col.replace('/µL)', ')')
        fixed_col = fixed_col.replace('µL)', ')') 
        
        # Clean up any remaining double µL patterns
        while 'µLµL' in fixed_col:
            fixed_col = fixed_col.replace('µLµL', 'µL')
        
        # Fix missing µL in column headers
        if '/μL' in fixed_col:
            fixed_col = fixed_col.replace('/μL', '/µL')
        if '(μL)' in fixed_col:
            fixed_col = fixed_col.replace('(μL)', '(µL)')
        if 'μL' in fixed_col and 'µL' not in fixed_col:
            fixed_col = fixed_col.replace('μL', 'µL')
        
        # Specific patterns for medical data
        if 'PLT (' in fixed_col and fixed_col.endswith('/'):
            fixed_col = fixed_col.replace('PLT (×10³/', 'PLT (×10³/µL')
            
        new_columns.append(fixed_col)
    
    # Apply the fixed column names
    df.columns = pd.Index(new_columns)
    
    # Check for and resolve duplicate column names
    if df.columns.duplicated().any():
        # Create a new list of column names
        new_columns = []
        seen_columns = {}
        
        for col in df.columns:
            if col in seen_columns:
                # If we've seen this column name before, add a numeric suffix
                seen_columns[col] += 1
                new_columns.append(f"{col}_{seen_columns[col]}")
            else:
                # First time seeing this column name
                seen_columns[col] = 0
                new_columns.append(col)
        
        # Assign the new column names to the dataframe
        df.columns = new_columns
    
    # Reset index to start from 1 instead of 0
    df.index = df.index + 1
    
    return df, filename

def get_descriptive_stats(df, columns=None):
    """
    Calculate descriptive statistics for specified columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    columns : list or None
        List of column names to analyze. If None, all columns will be analyzed.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe containing the descriptive statistics
    """
    # Initialize stats dictionary
    stats = {}
    
    # If no columns specified, use all columns
    if columns is None:
        columns = df.columns.tolist()
    
    # Print DataFrame information for debugging
    print(f"DataFrame shape in get_descriptive_stats: {df.shape}")
    
    for col in columns:
        # Skip if column doesn't exist in the DataFrame
        if col not in df.columns:
            print(f"Warning: Column {col} not found in DataFrame")
            continue
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                # Print raw column values for debugging
                values = df[col].dropna().values
                if len(values) > 0:
                    print(f"Column {col} values: min={values.min()}, max={values.max()}, mean={values.mean():.3f}")
                
                # Skip statistics calculation if there are no valid values
                if df[col].count() == 0:
                    col_stats = {
                        'Count': 0,
                        'Missing': df[col].isnull().sum(),
                        'Mean': 'NA',
                        'Median': 'NA',
                        'Mode': 'NA',
                        'Std Dev': 'NA',
                        'Mean \u00B1 Std Dev': 'NA',
                        'Variance': 'NA',
                        'Min': 'NA',
                        'Max': 'NA',
                        '25%': 'NA',
                        '50%': 'NA',
                        '75%': 'NA',
                        'Skewness': 'NA',
                        'Kurtosis': 'NA'
                    }
                else:
                    # Calculate mean and standard deviation from raw values to ensure accuracy
                    values = df[col].dropna().values
                    mean_value = values.mean() if len(values) > 0 else np.nan
                    std_value = values.std() if len(values) > 0 else np.nan
                    min_value = values.min() if len(values) > 0 else np.nan
                    max_value = values.max() if len(values) > 0 else np.nan
                    
                    # Handle NaN values in calculations
                    if np.isnan(mean_value) or np.isnan(std_value):
                        mean_std_format = 'NA'
                        mean_value_display = 'NA'
                        std_value_display = 'NA'
                    else:
                        # Format mean and std values individually with 2 decimal places
                        mean_formatted = f"{mean_value:.2f}"
                        std_formatted = f"{std_value:.2f}"
                        
                        # Store the formatted string representation 
                        # Use the Unicode plus-minus symbol ±
                        mean_std_format = f"{mean_formatted} \u00B1 {std_formatted}"
                        mean_value_display = round(mean_value, 3)
                        std_value_display = round(std_value, 3)
                    
                    # Get mode safely
                    try:
                        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'NA'
                        if not isinstance(mode_value, str) and not np.isnan(mode_value):
                            mode_value = round(mode_value, 3)
                    except:
                        mode_value = 'NA'
                    
                    # Get other quantile values safely
                    try:
                        median = np.median(values) if len(values) > 0 else np.nan
                        median_display = round(median, 3) if not np.isnan(median) else 'NA'
                    except:
                        median_display = 'NA'
                    
                    try:
                        variance = np.var(values) if len(values) > 0 else np.nan
                        variance_display = round(variance, 3) if not np.isnan(variance) else 'NA'
                    except:
                        variance_display = 'NA'
                    
                    try:
                        # Make sure we're getting min value directly from numpy array
                        min_val = np.min(values) if len(values) > 0 else np.nan
                        # Print debug info
                        print(f"Min value for {col}: raw={min_val}")
                    except Exception as e:
                        print(f"Error calculating min for {col}: {str(e)}")
                        min_val = np.nan
                    
                    try:
                        # Make sure we're getting max value directly from numpy array
                        max_val = np.max(values) if len(values) > 0 else np.nan
                        # Print debug info
                        print(f"Max value for {col}: raw={max_val}")
                    except Exception as e:
                        print(f"Error calculating max for {col}: {str(e)}")
                        max_val = np.nan
                    
                    try:
                        q25 = np.percentile(values, 25) if len(values) > 0 else np.nan
                        q25_display = round(q25, 3) if not np.isnan(q25) else 'NA'
                    except:
                        q25_display = 'NA'
                    
                    try:
                        q50 = np.percentile(values, 50) if len(values) > 0 else np.nan
                        q50_display = round(q50, 3) if not np.isnan(q50) else 'NA'
                    except:
                        q50_display = 'NA'
                    
                    try:
                        q75 = np.percentile(values, 75) if len(values) > 0 else np.nan
                        q75_display = round(q75, 3) if not np.isnan(q75) else 'NA'
                    except:
                        q75_display = 'NA'
                    
                    try:
                        skew = df[col].skew() if df[col].count() > 2 else np.nan
                        skew_display = round(skew, 3) if not np.isnan(skew) else 'NA'
                    except:
                        skew_display = 'NA'
                    
                    try:
                        kurt = df[col].kurtosis() if df[col].count() > 3 else np.nan
                        kurt_display = round(kurt, 3) if not np.isnan(kurt) else 'NA'
                    except:
                        kurt_display = 'NA'
                    
                    # Format min and max values consistently for display
                    # For integers or floats, don't force rounding to 3 decimal places
                    if isinstance(min_val, np.integer) or (isinstance(min_val, int) and not np.isnan(min_val)):
                        min_display = int(min_val)  # Show integers without decimal points
                    elif isinstance(min_val, (float, np.floating)) and not np.isnan(min_val):
                        min_display = round(min_val, 3)  # Round floats to 3 decimal places
                    else:
                        min_display = 'NA'
                        
                    if isinstance(max_val, np.integer) or (isinstance(max_val, int) and not np.isnan(max_val)):
                        max_display = int(max_val)  # Show integers without decimal points
                    elif isinstance(max_val, (float, np.floating)) and not np.isnan(max_val):
                        max_display = round(max_val, 3)  # Round floats to 3 decimal places
                    else:
                        max_display = 'NA'
                    
                    print(f"Final min/max values for {col}: min={min_display}, max={max_display}")
                    
                    # Format all float values to have at most 3 decimal places
                    col_stats = {
                        'Count': df[col].count(),
                        'Missing': df[col].isnull().sum(),
                        'Mean': mean_value_display,
                        'Median': median_display,
                        'Mode': mode_value,
                        'Std Dev': std_value_display,
                        'Mean \u00B1 Std Dev': mean_std_format,
                        'Variance': variance_display,
                        'Min': min_display,
                        'Max': max_display,
                        '25%': q25_display,
                        '50%': q50_display,
                        '75%': q75_display,
                        'Skewness': skew_display,
                        'Kurtosis': kurt_display
                    }
            except Exception as e:
                # If any error occurs, return NA for numerical statistics
                col_stats = {
                    'Count': df[col].count(),
                    'Missing': df[col].isnull().sum(),
                    'Mean': 'NA',
                    'Median': 'NA',
                    'Mode': 'NA',
                    'Std Dev': 'NA',
                    'Mean \u00B1 Std Dev': 'NA',
                    'Variance': 'NA',
                    'Min': 'NA',
                    'Max': 'NA',
                    '25%': 'NA',
                    '50%': 'NA',
                    '75%': 'NA',
                    'Skewness': 'NA',
                    'Kurtosis': 'NA'
                }
            stats[col] = col_stats
        else:
            # For non-numeric columns, calculate categorical statistics
            try:
                n_unique = df[col].nunique()
                
                # Handle case where column is entirely empty or has no values
                if df[col].count() == 0 or df[col].value_counts().empty:
                    col_stats = {
                        'Count': 0,
                        'Missing': df[col].isnull().sum(),
                        'Unique Values': 0,
                        'Mode': 'NA',
                        'Mode Frequency': 'NA',
                        'Mode Percentage': 'NA'
                    }
                else:
                    # Get mode safely
                    try:
                        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'NA'
                    except:
                        mode_value = 'NA'
                    
                    # Get mode frequency safely
                    try:
                        mode_freq = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 'NA'
                    except:
                        mode_freq = 'NA'
                    
                    # Format the mode percentage to have a maximum of 2 decimal places
                    try:
                        if df[col].count() > 0 and not df[col].value_counts().empty:
                            mode_pct = (df[col].value_counts().iloc[0] / df[col].count() * 100)
                            mode_pct_rounded = round(mode_pct, 2)  # Round to 2 decimals
                        else:
                            mode_pct_rounded = 'NA'
                    except:
                        mode_pct_rounded = 'NA'
                    
                    col_stats = {
                        'Count': df[col].count(),
                        'Missing': df[col].isnull().sum(),
                        'Mean': 'NA', 
                        'Median': 'NA',
                        'Mode': mode_value,
                        'Std Dev': 'NA',
                        'Mean \u00B1 Std Dev': 'NA',
                        'Variance': 'NA',
                        'Min': 'NA',
                        'Max': 'NA',
                        '25%': 'NA',
                        '50%': 'NA',
                        '75%': 'NA',
                        'Skewness': 'NA',
                        'Kurtosis': 'NA',
                        'Unique Values': n_unique,
                        'Mode Frequency': mode_freq,
                        'Mode Percentage': mode_pct_rounded
                    }
            except Exception as e:
                # Handle errors in categorical statistics calculation
                col_stats = {
                    'Count': df[col].count() if not pd.isna(df[col].count()) else 0,
                    'Missing': df[col].isnull().sum() if not pd.isna(df[col].isnull().sum()) else 0,
                    'Mean': 'NA', 
                    'Median': 'NA',
                    'Mode': 'NA',
                    'Std Dev': 'NA',
                    'Mean \u00B1 Std Dev': 'NA',
                    'Variance': 'NA',
                    'Min': 'NA',
                    'Max': 'NA',
                    '25%': 'NA',
                    '50%': 'NA',
                    '75%': 'NA',
                    'Skewness': 'NA',
                    'Kurtosis': 'NA',
                    'Unique Values': 'NA',
                    'Mode Frequency': 'NA',
                    'Mode Percentage': 'NA'
                }
            stats[col] = col_stats
    
    # Convert to dataframe
    stats_df = pd.DataFrame(stats)
    
    # Convert all values to strings to prevent Arrow conversion errors
    # This is especially important for 'Mean ± Std Dev' which can cause issues
    for col in stats_df.columns:
        # Check if Mean ± Std Dev is already a string for this column
        if 'Mean \u00B1 Std Dev' in stats_df.index and isinstance(stats_df.loc['Mean \u00B1 Std Dev', col], str):
            # Convert the entire column to strings to ensure consistent display
            stats_df[col] = stats_df[col].astype(str)
    
    return stats_df

def infer_data_types(df):
    """
    Infer data types for each column in the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
        
    Returns:
    --------
    dict
        Dictionary mapping column names to inferred data types
    """
    data_types = {}
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's likely an integer
            if df[col].dropna().apply(lambda x: x == int(x)).all():
                # Check if it's likely categorical
                if df[col].nunique() / len(df) < 0.05 or df[col].nunique() < 10:
                    data_types[col] = 'categorical'
                else:
                    data_types[col] = 'integer'
            else:
                data_types[col] = 'continuous'
        elif pd.api.types.is_datetime64_dtype(df[col]):
            data_types[col] = 'datetime'
        else:
            # Check if it's a categorical variable
            if df[col].nunique() / len(df) < 0.05 or df[col].nunique() < 10:
                data_types[col] = 'categorical'
            else:
                data_types[col] = 'text'
    
    return data_types

def generate_age_groups(df, age_column, age_type='years', age_group_col_name='Generated Age Group'):
    """
    Generate age groups based on the age data
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    age_column : str
        The column name containing age data
    age_type : str
        Type of age data ('years', 'months', 'days')
    age_group_col_name : str
        The name to use for the age group column (default: 'Generated Age Group')
        
    Returns:
    --------
    pd.DataFrame, dict, str
        DataFrame with age groups added, grouping info dictionary, and the actual column name used
    """
    if not pd.api.types.is_numeric_dtype(df[age_column]):
        raise ValueError(f"Age column '{age_column}' is not numeric")
    
    # Create a copy to avoid modifying the original dataframe
    df_with_groups = df.copy()
    
    # Get unique age values and sort them
    unique_ages = sorted(df[age_column].unique())
    print(f"Unique ages in dataset: {unique_ages}")
    
    # Get min and max age
    min_age = min(unique_ages)
    max_age = max(unique_ages)
    print(f"Min age: {min_age}, Max age: {max_age}")
    
    # Initialize age group column
    final_col_name = age_group_col_name
    if age_group_col_name in df.columns:
        # If the column already exists, add a suffix to make it unique
        counter = 1
        while f"{age_group_col_name}_{counter}" in df.columns:
            counter += 1
        final_col_name = f"{age_group_col_name}_{counter}"
    
    # Simple function to generate age group for each row directly
    def assign_age_group(age):
        if age_type == 'years':
            if age < 18:  # Children
                # 1-year groups for children
                start = (age // 1) * 1
                end = start + 1 - 1
                return f"{start}-{end} yrs"
            else:  # Adults
                # 5-year groups for adults
                start = (age // 5) * 5
                end = start + 5 - 1
                return f"{start}-{end} yrs"
        elif age_type == 'months':
            if age <= 24:  # Young children
                # 1-month groups for infants
                start = (age // 1) * 1
                end = start + 1 - 1
                return f"{start}-{end} mo"
            else:  # Older children in months
                # 3-month groups
                start = (age // 3) * 3
                end = start + 3 - 1
                return f"{start}-{end} mo"
        elif age_type == 'days':
            if age <= 90:  # Neonates
                # 7-day groups
                start = (age // 7) * 7
                end = start + 7 - 1
                return f"{start}-{end} days"
            else:  # Older infants
                # Convert to weeks
                weeks = age // 7
                start = weeks
                end = start + 1 - 1
                return f"{start}-{end} wk"
        
        # Default fallback for unexpected cases
        return f"Age: {age}"
    
    # Apply the function to each row
    df_with_groups[final_col_name] = df[age_column].apply(assign_age_group)
    
    # Get the age groups for debugging
    age_groups = df_with_groups[final_col_name].unique()
    print(f"Generated age groups: {sorted(list(age_groups))}")
    
    # Insert the age group column right after the age column
    age_col_idx = df.columns.get_loc(age_column)
    col_list = list(df.columns)
    col_list.insert(age_col_idx + 1, final_col_name)
    
    # Reorder columns
    df_with_groups = df_with_groups[col_list]
    
    # For consistency with the original function, create a grouping_info dictionary
    # Determine bin size based on age type
    if age_type == 'years' and min_age >= 18:
        bin_size = 5
    elif age_type == 'years':
        bin_size = 1
    elif age_type == 'months' and max_age <= 24:
        bin_size = 1
    elif age_type == 'months':
        bin_size = 3
    elif age_type == 'days' and max_age <= 90:
        bin_size = 7
    else:
        bin_size = 7  # 1 week
    
    grouping_info = {
        'age_type': age_type,
        'bin_size': bin_size,
        'min_age': min_age,
        'max_age': max_age,
        'age_groups': sorted(list(age_groups))
    }
    
    return df_with_groups, grouping_info, final_col_name

def analyze_master_parameters(df, master_columns, analysis_columns=None):
    """
    Analyze data grouped by master parameters
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    master_columns : list
        List of master column names to use for grouping
    analysis_columns : list or None
        List of column names to analyze within each group.
        If None, all columns will be analyzed.
        
    Returns:
    --------
    dict
        Dictionary of master parameter analysis results
    """
    if not master_columns:
        return None
    
    # If no analysis columns specified, use all columns except master columns
    if analysis_columns is None:
        analysis_columns = [col for col in df.columns if col not in master_columns]
    
    results = {}
    
    # For each master column, identify column type and create groupings
    for master_col in master_columns:
        try:
            # Skip if column doesn't exist
            if master_col not in df.columns:
                continue
                
            # Determine column type
            col_type = "categorical"
            if pd.api.types.is_numeric_dtype(df[master_col]):
                col_type = "numerical"
                
            col_results = {
                'type': col_type,
                'groups': {}
            }
            
            if col_type == "categorical":
                # For categorical master columns, group by unique values
                groups = df[master_col].dropna().unique()
                
                for group in groups:
                    # Filter data for this group
                    group_df = df[df[master_col] == group]
                    
                    # Calculate statistics for each analysis column within this group
                    # For age columns, ensure the 'Generated Age Group' is properly calculated
                    # This ensures consistent age groups in master parameter analysis
                    if 'Age' in group_df.columns and 'Generated Age Group' in group_df.columns:
                        # Make sure the age groups are correctly assigned
                        # This is necessary to fix the issues with ages like 60
                        for idx, row in group_df.iterrows():
                            age = row['Age']
                            if pd.api.types.is_numeric_dtype(pd.Series([age])):
                                # For adults, use 5-year groups
                                if age >= 18:
                                    start = (age // 5) * 5
                                    end = start + 5 - 1
                                    age_group = f"{start}-{end} yrs"
                                    # Update the age group for this row
                                    group_df.at[idx, 'Generated Age Group'] = age_group
                    
                    # Create a copy of the group dataframe to avoid modifying the original
                    group_df_copy = group_df.copy()
                    
                    # Calculate statistics for this group
                    # Calculate raw statistics first for debugging
                    num_cols = [col for col in analysis_columns if pd.api.types.is_numeric_dtype(group_df_copy[col])]
                    raw_stats = {}
                    for col in num_cols:
                        if not group_df_copy[col].empty:
                            raw_stats[col] = {
                                'min': group_df_copy[col].min(),
                                'max': group_df_copy[col].max(),
                                'mean': group_df_copy[col].mean()
                            }
                    
                    # Print debug info for this group
                    print(f"Group '{group}' raw stats: {raw_stats}")
                    
                    # Now calculate the full statistics with the corrected data
                    group_stats = get_descriptive_stats(group_df_copy, analysis_columns)
                    
                    # Store results
                    col_results['groups'][str(group)] = {
                        'count': len(group_df_copy),
                        'stats': group_stats,
                        'raw_stats': raw_stats  # Include raw stats for debugging
                    }
            
            else:  # numerical column
                # For numerical columns, create bins/ranges
                # First try to infer if it's a discrete or continuous variable
                values = df[master_col].dropna().unique()
                
                if len(values) <= 10 or all(float(v).is_integer() for v in values if not pd.isna(v)):
                    # Discrete values - group by each unique value
                    for val in values:
                        # Filter data for this value
                        group_df = df[df[master_col] == val]
                        
                        # Calculate statistics for each analysis column within this group
                        group_stats = get_descriptive_stats(group_df, analysis_columns)
                        
                        # Store results
                        col_results['groups'][str(val)] = {
                            'count': len(group_df),
                            'stats': group_stats
                        }
                else:
                    # Continuous values - create bins
                    min_val = df[master_col].min()
                    max_val = df[master_col].max()
                    
                    # Create 5 bins
                    bins = pd.cut(df[master_col], bins=5)
                    bin_df = pd.DataFrame({master_col + '_bin': bins, 'original_index': df.index})
                    
                    # Group by bins
                    for bin_name, indices in bin_df.groupby(master_col + '_bin')['original_index']:
                        # Filter data for this bin
                        group_df = df.loc[indices]
                        
                        # Calculate statistics for each analysis column within this bin
                        group_stats = get_descriptive_stats(group_df, analysis_columns)
                        
                        # Store results with formatted bin name
                        # We need to handle the bin name format carefully
                        try:
                            # For pandas Interval objects
                            if hasattr(bin_name, 'left') and hasattr(bin_name, 'right'):
                                bin_label = f"{bin_name.left:.1f} - {bin_name.right:.1f}"
                            else:
                                # For other types, just convert to string
                                bin_label = str(bin_name)
                            
                            col_results['groups'][bin_label] = {
                                'count': len(group_df),
                                'stats': group_stats
                            }
                        except Exception as e:
                            # Fallback to simple string if there's an error formatting the bin
                            bin_label = str(bin_name)
                            col_results['groups'][bin_label] = {
                                'count': len(group_df),
                                'stats': group_stats
                            }
            
            # Store results for this master column
            results[master_col] = col_results
            
        except Exception as e:
            # Skip this master column if there's an error
            results[master_col] = {
                'type': 'error',
                'error': str(e)
            }
    
    return results

def detect_outliers(df, column, method='iqr'):
    """
    Detect outliers in a numerical column
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    column : str
        The column name to analyze
    method : str
        Method to use for outlier detection ('iqr' or 'zscore')
        
    Returns:
    --------
    pd.Series, dict
        Boolean mask indicating outliers and outlier stats
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    outliers = None
    stats = {}
    
    if method == 'iqr':
        # IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        stats = {
            'method': 'IQR',
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'num_outliers': outliers.sum(),
            'percentage_outliers': outliers.mean() * 100
        }
    
    elif method == 'zscore':
        # Z-score method
        mean = df[column].mean()
        std = df[column].std()
        
        z_scores = (df[column] - mean) / std
        outliers = (z_scores.abs() > 3)
        
        stats = {
            'method': 'Z-score',
            'mean': mean,
            'std': std,
            'num_outliers': outliers.sum(),
            'percentage_outliers': outliers.mean() * 100
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers, stats

def calculate_manual_paired_differences(df, manual_pairs):
    """
    Calculate differences for manually selected pre/post pairs (post minus pre)
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    manual_pairs : list
        List of tuples (pre_col, post_col) representing manually selected pairs
        
    Returns:
    --------
    pd.DataFrame, dict
        Enhanced dataframe with difference columns and information about the pairs
    """
    df_enhanced = df.copy()
    pairs_info = {
        'pairs_processed': [],
        'difference_columns': [],
        'summary': {}
    }
    
    # Calculate differences for each manually selected pair
    for pre_col, post_col in manual_pairs:
        try:
            if pre_col in df.columns and post_col in df.columns:
                # Ensure both columns are numeric
                if pd.api.types.is_numeric_dtype(df[pre_col]) and pd.api.types.is_numeric_dtype(df[post_col]):
                    # Calculate difference (post minus pre)
                    difference = df_enhanced[post_col] - df_enhanced[pre_col]
                    
                    # Create descriptive column name for difference
                    # Extract base parameter name from pre or post column
                    base_name = pre_col.replace('Pre ', '').replace('Post ', '').replace('pre ', '').replace('post ', '')
                    base_name = base_name.replace('Pre', '').replace('Post', '').replace('pre', '').replace('post', '').strip()
                    
                    if base_name:
                        diff_col_name = f"{base_name} Difference"
                    else:
                        diff_col_name = f"{post_col} - {pre_col} Difference"
                    
                    # Check if this difference column already exists to prevent duplicates
                    if diff_col_name in df_enhanced.columns:
                        # Skip creating duplicate difference column
                        print(f"Difference column '{diff_col_name}' already exists, skipping duplicate creation.")
                        continue
                    
                    # Add difference column to dataframe
                    df_enhanced[diff_col_name] = difference
                    
                    # Store information about this pair
                    pair_info = {
                        'pre_column': pre_col,
                        'post_column': post_col,
                        'difference_column': diff_col_name,
                        'base_parameter': base_name,
                        'mean_difference': difference.mean(),
                        'std_difference': difference.std(),
                        'min_difference': difference.min(),
                        'max_difference': difference.max()
                    }
                    
                    pairs_info['pairs_processed'].append(pair_info)
                    pairs_info['difference_columns'].append(diff_col_name)
                    
        except Exception as e:
            continue
    
    # Create summary
    pairs_info['summary'] = {
        'total_pairs_processed': len(pairs_info['pairs_processed']),
        'total_difference_columns_added': len(pairs_info['difference_columns']),
        'original_columns': len(df.columns),
        'enhanced_columns': len(df_enhanced.columns)
    }
    
    return df_enhanced, pairs_info

def calculate_paired_differences(df):
    """
    DEPRECATED: This function is kept for backward compatibility but should not be used.
    Use calculate_manual_paired_differences instead for manual pair selection.
    """
    import pandas as pd
    import re
    
    df_enhanced = df.copy()
    pairs_info = {
        'pairs_found': [],
        'difference_columns': [],
        'summary': {}
    }
    
    columns = df.columns.tolist()
    
    # Common patterns for pre/post identification
    pre_patterns = [r'pre[\s_-]?', r'before[\s_-]?', r'baseline[\s_-]?', r'initial[\s_-]?']
    post_patterns = [r'post[\s_-]?', r'after[\s_-]?', r'final[\s_-]?', r'end[\s_-]?']
    
    # Find potential pairs
    potential_pairs = []
    
    for col in columns:
        col_lower = col.lower()
        
        # Check if this is a pre column
        for pre_pattern in pre_patterns:
            if re.search(pre_pattern, col_lower):
                # Extract the base parameter name
                base_name = re.sub(pre_pattern, '', col_lower).strip()
                base_name = re.sub(r'^[\s_-]+|[\s_-]+$', '', base_name)
                
                # Look for corresponding post column
                for post_col in columns:
                    post_col_lower = post_col.lower()
                    for post_pattern in post_patterns:
                        if re.search(post_pattern, post_col_lower):
                            post_base = re.sub(post_pattern, '', post_col_lower).strip()
                            post_base = re.sub(r'^[\s_-]+|[\s_-]+$', '', post_base)
                            
                            # Check if base names match
                            if base_name == post_base or (len(base_name) > 3 and base_name in post_base):
                                # Ensure both columns are numeric
                                if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[post_col]):
                                    potential_pairs.append((col, post_col, base_name))
    
    # Remove duplicates and calculate differences
    unique_pairs = []
    for pre_col, post_col, base_name in potential_pairs:
        # Check if this pair is already processed
        reverse_exists = any(pair[0] == post_col and pair[1] == pre_col for pair in unique_pairs)
        if not reverse_exists:
            unique_pairs.append((pre_col, post_col, base_name))
    
    # Calculate differences for each pair
    for pre_col, post_col, base_name in unique_pairs:
        try:
            # Calculate difference (post minus pre)
            difference = df_enhanced[post_col] - df_enhanced[pre_col]
            
            # Create descriptive column name for difference
            if base_name:
                diff_col_name = f"{base_name.title()} Difference"
            else:
                diff_col_name = f"Difference ({post_col} - {pre_col})"
            
            # Ensure unique column name
            counter = 1
            original_name = diff_col_name
            while diff_col_name in df_enhanced.columns:
                diff_col_name = f"{original_name}_{counter}"
                counter += 1
            
            # Add difference column to dataframe
            df_enhanced[diff_col_name] = difference
            
            # Store information about this pair
            pair_info = {
                'pre_column': pre_col,
                'post_column': post_col,
                'difference_column': diff_col_name,
                'base_parameter': base_name,
                'mean_difference': difference.mean(),
                'std_difference': difference.std(),
                'min_difference': difference.min(),
                'max_difference': difference.max()
            }
            
            pairs_info['pairs_found'].append(pair_info)
            pairs_info['difference_columns'].append(diff_col_name)
            
        except Exception as e:
            continue
    
    # Create summary
    pairs_info['summary'] = {
        'total_pairs_found': len(pairs_info['pairs_found']),
        'total_difference_columns_added': len(pairs_info['difference_columns']),
        'original_columns': len(df.columns),
        'enhanced_columns': len(df_enhanced.columns)
    }
    
    return df_enhanced, pairs_info
