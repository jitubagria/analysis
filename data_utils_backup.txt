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
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            # Try with different encodings
            try:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            except:
                # Try with different separators
                try:
                    df = pd.read_csv(uploaded_file, sep=';')
                except:
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
    
    # Convert column names to string to avoid issues
    df.columns = df.columns.astype(str)
    
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
    
    for col in columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
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
                    # Calculate mean and standard deviation
                    mean_value = df[col].mean()
                    std_value = df[col].std()
                    
                    # Handle NaN values in calculations
                    if pd.isna(mean_value) or pd.isna(std_value):
                        mean_std_format = 'NA'
                        mean_value_display = 'NA'
                        std_value_display = 'NA'
                    else:
                        # Format mean and std values individually with 2 decimal places (max 3)
                        mean_formatted = f"{mean_value:.2f}"
                        std_formatted = f"{std_value:.2f}"
                        
                        # Store the formatted string representation 
                        # Use the regular ASCII plus-minus symbol ±
                        mean_std_format = f"{mean_formatted} \u00B1 {std_formatted}"
                        mean_value_display = round(mean_value, 3)
                        std_value_display = round(std_value, 3)
                    
                    # Get mode safely
                    try:
                        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'NA'
                        if not isinstance(mode_value, str) and not pd.isna(mode_value):
                            mode_value = round(mode_value, 3)
                    except:
                        mode_value = 'NA'
                    
                    # Get other quantile values safely
                    try:
                        median = df[col].median()
                        median_display = round(median, 3) if not pd.isna(median) else 'NA'
                    except:
                        median_display = 'NA'
                    
                    try:
                        variance = df[col].var()
                        variance_display = round(variance, 3) if not pd.isna(variance) else 'NA'
                    except:
                        variance_display = 'NA'
                    
                    try:
                        min_val = df[col].min()
                        min_display = round(min_val, 3) if isinstance(min_val, (float, int)) and not pd.isna(min_val) else min_val
                        if pd.isna(min_display):
                            min_display = 'NA'
                    except:
                        min_display = 'NA'
                    
                    try:
                        max_val = df[col].max()
                        max_display = round(max_val, 3) if isinstance(max_val, (float, int)) and not pd.isna(max_val) else max_val
                        if pd.isna(max_display):
                            max_display = 'NA'
                    except:
                        max_display = 'NA'
                    
                    try:
                        q25 = df[col].quantile(0.25)
                        q25_display = round(q25, 3) if not pd.isna(q25) else 'NA'
                    except:
                        q25_display = 'NA'
                    
                    try:
                        q50 = df[col].quantile(0.50)
                        q50_display = round(q50, 3) if not pd.isna(q50) else 'NA'
                    except:
                        q50_display = 'NA'
                    
                    try:
                        q75 = df[col].quantile(0.75)
                        q75_display = round(q75, 3) if not pd.isna(q75) else 'NA'
                    except:
                        q75_display = 'NA'
                    
                    try:
                        skew = df[col].skew()
                        skew_display = round(skew, 3) if not pd.isna(skew) else 'NA'
                    except:
                        skew_display = 'NA'
                    
                    try:
                        kurt = df[col].kurtosis()
                        kurt_display = round(kurt, 3) if not pd.isna(kurt) else 'NA'
                    except:
                        kurt_display = 'NA'
                    
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
    
    ages = df[age_column]
    min_age = ages.min()
    max_age = ages.max()
    
    # Determine grouping logic based on age range and type
    if age_type == 'months':
        if max_age <= 24:
            # For infants/toddlers (monthly)
            scale = 'months'
            bin_size = 1
            max_val = int(np.ceil(max_age)) + 1
            bins = list(range(0, max_val + bin_size, bin_size))
            labels = [f"{i}-{i+bin_size - 1} mo" for i in bins[:-1]]
        else:
            # For older children in months
            scale = 'months'
            bin_size = 3
            max_val = int(np.ceil(max_age)) + bin_size
            bins = list(range(0, max_val, bin_size))
            labels = [f"{i}-{i+bin_size - 1} mo" for i in bins[:-1]]
            
    elif age_type == 'days':
        if max_age <= 90:
            # For neonates/young infants (daily)
            scale = 'days'
            bin_size = 7
            max_val = int(np.ceil(max_age)) + bin_size
            bins = list(range(0, max_val, bin_size))
            labels = [f"{i}-{i+bin_size - 1} days" for i in bins[:-1]]
        else:
            # For older infants in days, convert to weeks
            scale = 'weeks'
            bin_size = 7
            max_val = int(np.ceil(max_age/7)) * 7 + bin_size
            bins = list(range(0, max_val, bin_size))
            labels = [f"{int(i/7)}-{int((i+bin_size - 1)/7)} wk" for i in bins[:-1]]
            
    else:  # years
        if min_age >= 18:
            # For adults (5-year groups)
            scale = 'adult_5yrs'
            bin_size = 5
            min_val = (min_age // bin_size) * bin_size
            max_val = int(np.ceil(max_age / bin_size)) * bin_size + bin_size
            bins = list(range(min_val, max_val, bin_size))
            labels = [f"{i}-{i+bin_size - 1} yrs" for i in bins[:-1]]
        elif max_age <= 18:
            # For children/adolescents (1-year groups)
            scale = 'child_1yr'
            bin_size = 1
            max_val = int(np.ceil(max_age)) + bin_size
            bins = list(range(0, max_val, bin_size))
            labels = [f"{i}-{i+bin_size - 1} yrs" for i in bins[:-1]]
        else:
            # For general population (10-year groups)
            scale = 'general_10yrs'
            bin_size = 10
            max_val = int(np.ceil(max_age / bin_size)) * bin_size + bin_size
            bins = list(range(0, max_val, bin_size))
            labels = [f"{i}-{i+bin_size - 1} yrs" for i in bins[:-1]]
    
    # Assign age groups
    age_groups = pd.cut(ages, bins=bins, labels=labels, right=False)
    
    # Make a copy of the dataframe to avoid modifying the original
    df_with_groups = df.copy()
    
    # Check if we need to use a different column name to avoid conflicts
    final_col_name = age_group_col_name
    if age_group_col_name in df.columns:
        # If the column already exists, add a suffix to make it unique
        counter = 1
        while f"{age_group_col_name}_{counter}" in df.columns:
            counter += 1
        final_col_name = f"{age_group_col_name}_{counter}"
    
    # Get the column position of the age column
    age_col_idx = df.columns.get_loc(age_column)
    
    # Insert the age group column right after the age column
    col_list = list(df.columns)
    col_list.insert(age_col_idx + 1, final_col_name)
    
    # Create the new dataframe with the age group column in the right position
    df_with_groups = df.copy()
    df_with_groups[final_col_name] = age_groups
    df_with_groups = df_with_groups[col_list]
    
    grouping_info = {
        'scale': scale,
        'bin_size': bin_size,
        'min_age': min_age,
        'max_age': max_age,
        'bins': bins,
        'labels': labels
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
                    group_stats = get_descriptive_stats(group_df, analysis_columns)
                    
                    # Store results
                    col_results['groups'][str(group)] = {
                        'count': len(group_df),
                        'stats': group_stats
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
