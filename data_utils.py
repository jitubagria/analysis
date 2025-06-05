import pandas as pd
import numpy as np
import io
import re


def load_data(uploaded_file):
    """
    Load data from uploaded file (CSV, Excel)
<<<<<<< Updated upstream

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
    df = None  # Initialize df
=======
    """
    filename = uploaded_file.name
    file_ext = filename.split('.')[-1].lower()
>>>>>>> Stashed changes

    if file_ext == 'csv':
        # Using the CSV loading logic from your provided file
        try:
<<<<<<< Updated upstream
            # First try with pandas' automatic delimiter detection
            uploaded_file.seek(0)  # Ensure pointer is at the beginning
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except UnicodeDecodeError as e_unicode_initial:
            # Try with different encodings and delimiter detection
            # Added utf-8 here too
            encodings_to_try = ['latin1', 'cp1252',
                                'iso-8859-1', 'utf-16', 'utf-8']
            separators_to_try = [None, ',', ';', '\t', ' ', '|']
            last_exception_csv = e_unicode_initial

            for encoding in encodings_to_try:
                if df is not None and df.shape[1] > 1:
                    break
                for sep in separators_to_try:
                    try:
                        uploaded_file.seek(0)
                        current_df = pd.read_csv(
                            uploaded_file,
                            encoding=encoding,
                            sep=sep if sep is not None else None,  # sep=None for auto with python engine
                            engine='python' if sep is None else pd.io.common.get_engine(
                                'c')  # Default C engine if sep specified
                        )
                        if current_df.shape[1] > 1:
                            df = current_df
                            print(
                                f"Successfully loaded CSV with encoding: {encoding}, separator: {repr(sep)}")
                            last_exception_csv = None
                            break
                        else:
                            last_exception_csv = Exception(
                                f"Loaded with 1 column using encoding: {encoding}, separator: {repr(sep)}")
                    except Exception as e_inner:
                        last_exception_csv = e_inner  # Keep track of the latest error
                if df is not None and df.shape[1] > 1:
                    break

            if df is None or df.shape[1] == 1:
                raise Exception(
                    f"Failed to load CSV with proper column separation after trying multiple options. Last error: {str(last_exception_csv)}")
        except Exception as e_outer:  # Catch other potential errors from initial try
            # Final fallback: try different separators with default encoding (usually utf-8)
            separators_to_try = [',', ';', '\t', ' ', '|']
            last_exception_csv = e_outer
            df = None  # Reset df

            for sep in separators_to_try:
                try:
                    uploaded_file.seek(0)
                    current_df = pd.read_csv(uploaded_file, sep=sep)
                    if current_df.shape[1] > 1:
                        df = current_df
                        print(
                            f"Successfully loaded CSV with default encoding and separator: {repr(sep)}")
                        last_exception_csv = None
                        break
                    else:
                        last_exception_csv = Exception(
                            f"Loaded with 1 column using default encoding and separator: {repr(sep)}")
                except Exception as e_fallback:
                    last_exception_csv = e_fallback

            if df is None or df.shape[1] == 1:
                raise Exception(
                    f"Failed to load CSV after all fallbacks. Last error: {str(last_exception_csv)}")

    elif file_ext in ['xlsx', 'xls']:
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(
                uploaded_file,
                engine='openpyxl',
                keep_default_na=True,
                na_values=['NA']
            )
=======
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except UnicodeDecodeError as e:
            encodings_to_try = ['latin1', 'cp1252', 'iso-8859-1', 'utf-16']
            separators_to_try = [None, ',', ';', '\t', ' ', '|']
            df = None
            for encoding in encodings_to_try:
                for sep in separators_to_try:
                    try:
                        uploaded_file.seek(0)
                        if sep is None:
                            df = pd.read_csv(
                                uploaded_file, encoding=encoding, sep=None, engine='python')
                        else:
                            df = pd.read_csv(
                                uploaded_file, encoding=encoding, sep=sep)
                        if df.shape[1] > 1:
                            break
                    except:
                        continue
                if df is not None and df.shape[1] > 1:
                    break
            if df is None or df.shape[1] == 1:
                raise Exception(
                    f"Failed to load CSV with proper column separation. Encoding error: {str(e)}")
        except Exception as e:
            separators_to_try = [',', ';', '\t', ' ', '|']
            df = None
            for sep in separators_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=sep)
                    if df.shape[1] > 1:
                        break
                except:
                    continue
            if df is None:
                raise Exception(f"Failed to load CSV: {str(e)}")

    elif file_ext in ['xlsx', 'xls']:
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl',
                               keep_default_na=True, na_values=['NA'])
>>>>>>> Stashed changes
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
        except Exception as e:
            raise Exception(f"Failed to load Excel file: {str(e)}")
    else:
        raise Exception(f"Unsupported file format: {file_ext}")

<<<<<<< Updated upstream
    if df is None:
        raise Exception("DataFrame could not be loaded.")

    df = df.replace(r'^\s*$', np.nan, regex=True)
    df.columns = df.columns.astype(str)

    # --- Updated clean_header function ---

    def clean_header(header_name):
        header_name = str(header_name)  # Ensure header_name is a string

        # Define the standard clean internal unit string we want for PLT.
        # This uses 'x', '^3', and crucially, 'µL'.
        # It does NOT add surrounding parentheses itself to prevent nesting.
        standard_plt_unit_core = "x10^3/µL"

        # --- Specific replacements for PLT-related units ---
        # Order is important: replace more specific/longer patterns first.

        # Case 1: Corrupted unit "×⃞10³/µL" (box char, and µL present) -> standard form
        header_name = header_name.replace("×⃞10³/µL", standard_plt_unit_core)

        # Case 2: Unit "×10³/µL" (already has µL, but uses '×') -> standard form
        header_name = header_name.replace("×10³/µL", standard_plt_unit_core)

        # Case 3: Corrupted unit "×⃞10³/" (box char, slash present, µL might be missing/implied) -> standard form
        # This will add the standard_plt_unit_core which includes /µL
        header_name = header_name.replace("×⃞10³/", standard_plt_unit_core)

        # Case 4: Normalize "x10³/µL" (if it used 'x' but not '^3' or had other minor variations)
        # This ensures if the unit is already mostly clean, it conforms to the exact standard_plt_unit_core.
        # This is useful if the source CSV might sometimes have "x10³/µL" directly.
        header_name = header_name.replace("x10³/µL", standard_plt_unit_core)

        # --- General character cleanups that apply to the whole header string ---
        # These apply after specific unit string transformations.
        header_name = header_name.replace("×", "x")
        header_name = header_name.replace("³", "^3")

        # Ensure "µL" is NOT replaced by "uL" or an empty string.
        # The line `header_name = header_name.replace("µL", "uL")` has been intentionally omitted
        # to preserve the "µL" symbol as per your request.

        # Remove any remaining box/placeholder characters anywhere in the header.
        header_name = re.sub(r'[□⃞]', '', header_name)

        return header_name.strip()
    # --- End of clean_header function ---

    df.columns = pd.Index([clean_header(col) for col in df.columns])

    if df.columns.duplicated().any():
        new_columns = []
        seen_columns = {}
        for col in df.columns:
            col_str = str(col)
            if col_str in seen_columns:
                seen_columns[col_str] += 1
                new_columns.append(f"{col_str}_{seen_columns[col_str]}")
            else:
                seen_columns[col_str] = 0
                new_columns.append(col_str)
        df.columns = new_columns

    df.index = df.index + 1

    return df, filename


# --- Remaining functions from your provided file (get_descriptive_stats, infer_data_types, etc.) ---
# --- These are kept as you provided them in the uploaded file ---

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
    stats = {}

    if columns is None:
        columns = df.columns.tolist()

    print(f"DataFrame shape in get_descriptive_stats: {df.shape}")

=======
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df.columns = df.columns.astype(str)

    new_columns = []
    for col in df.columns:
        fixed_col = str(col)

        # --- START REVISED PRIORITY 1 SECTION ---
        # Priority 1: Handle specific misinterpretations that could change / to (
        # If original "/µL" was misread by pandas as "(ÂµL"
        # Check if it ends with (ÂµL but not " (ÂµL" (i.e. not a space before it, implying it's not like "Unit (ÂµL)")
        if fixed_col.endswith("(ÂµL") and not fixed_col.endswith(" (ÂµL"):
            if fixed_col == "(ÂµL":  # If the entire string is this misinterpretation
                fixed_col = "/µL"
            # If it's like "SomePrefix(ÂµL" where original was "SomePrefix/µL"
            else:
                fixed_col = fixed_col.replace("(ÂµL", "/µL")
        # If original "/μL" (Greek) was misread by pandas as "(Î¼L"
        elif fixed_col.endswith("(Î¼L") and not fixed_col.endswith(" (Î¼L"):
            if fixed_col == "(Î¼L":
                fixed_col = "/μL"  # Will be standardized to micro sign later
            else:
                fixed_col = fixed_col.replace("(Î¼L", "/μL")

        # Then handle other misinterpretations involving 'Âµ' or 'Î¼' for / and () contexts
        elif "/ÂµL" in fixed_col:
            fixed_col = fixed_col.replace("/ÂµL", "/µL")
        elif "(ÂµL)" in fixed_col:  # For genuine parenthesized cases like "Name (ÂµL)"
            # This correctly keeps the parentheses
            fixed_col = fixed_col.replace("(ÂµL)", "(µL)")

        elif "/Î¼L" in fixed_col:
            fixed_col = fixed_col.replace("/Î¼L", "/µL")
        elif "(Î¼L)" in fixed_col:  # For "Name (Î¼L)"
            fixed_col = fixed_col.replace("(Î¼L)", "(µL)")
        # --- END REVISED PRIORITY 1 SECTION ---

        # Priority 2: Standardize Greek mu (μ) to micro sign (µ) in specific / and () contexts
        if "/μL" in fixed_col:
            fixed_col = fixed_col.replace("/μL", "/µL")
        if "(μL)" in fixed_col:
            fixed_col = fixed_col.replace("(μL)", "(µL)")

        # Priority 3: Handle more general cases of 'ÂµL' or 'Î¼L' (that don't start with / or ()
        if "ÂµL" in fixed_col and not (fixed_col.startswith(("/", "("))):
            fixed_col = fixed_col.replace("ÂµL", "µL")
        if "Î¼L" in fixed_col and not (fixed_col.startswith(("/", "("))):
            fixed_col = fixed_col.replace("Î¼L", "µL")

        # Priority 4: General standardization of Greek mu (μL) to micro sign (µL)
        if "μL" in fixed_col and "µL" not in fixed_col:
            fixed_col = fixed_col.replace("μL", "µL")

        # Original specific pattern fixes
        fixed_col = fixed_col.replace('Ã10Â³/Â', '×10³/µL')
        fixed_col = fixed_col.replace('Ã10Â³/', '×10³/µL')

        fixed_col = fixed_col.replace('Ã', '×')

        fixed_col = fixed_col.replace('Â³', '³')
        fixed_col = fixed_col.replace('Â²', '²')

        # General Â removal - runs AFTER specific "ÂµL" patterns are handled.
        if "µL" not in fixed_col and "×10³/" not in fixed_col:
            fixed_col = fixed_col.replace('Â', '')

        if '×10³/' in fixed_col and not fixed_col.endswith('µL') and 'µL' not in fixed_col:
            fixed_col = fixed_col.replace('×10³/', '×10³/µL')

        fixed_col = fixed_col.replace('µLµL', 'µL')

        while 'µLµL' in fixed_col:
            fixed_col = fixed_col.replace('µLµL', 'µL')

        if 'PLT (' in fixed_col and fixed_col.endswith('/'):
            if '×10³/' in fixed_col:
                fixed_col = fixed_col.replace('×10³/', '×10³/µL')

        new_columns.append(fixed_col)

    df.columns = pd.Index(new_columns)

    if df.columns.duplicated().any():
        new_columns_dedup = []
        seen_columns = {}
        for col in df.columns:
            if col in seen_columns:
                seen_columns[col] += 1
                new_columns_dedup.append(f"{col}_{seen_columns[col]}")
            else:
                seen_columns[col] = 0
                new_columns_dedup.append(col)
        df.columns = pd.Index(new_columns_dedup)

    df.index = df.index + 1
    return df, filename


def get_descriptive_stats(df, columns=None):
    stats = {}
    if columns is None:
        columns = df.columns.tolist()

>>>>>>> Stashed changes
    for col in columns:
        if col not in df.columns:
            continue

<<<<<<< Updated upstream
        col_data = df[col]
        col_stats_dict = {
            'Count': col_data.count(),
            'Missing': col_data.isnull().sum(),
            'Mean': 'NA', 'Median': 'NA', 'Mode': 'NA', 'Std Dev': 'NA',
            'Mean ± Std Dev': 'NA', 'Variance': 'NA',
            'Min': 'NA', 'Max': 'NA',
            '25%': 'NA', '50%': 'NA', '75%': 'NA',
            'Skewness': 'NA', 'Kurtosis': 'NA',
            'Unique Values': 'NA', 'Mode Frequency': 'NA', 'Mode Percentage': 'NA'
        }

        if pd.api.types.is_numeric_dtype(col_data):
            if col_stats_dict['Count'] > 0:
                values = col_data.dropna().values

                mean_value = np.mean(values)
                median_value = np.median(values)
                std_value = np.std(values)
                var_value = np.var(values)
                min_value = np.min(values)
                max_value = np.max(values)
                q25_value = np.percentile(values, 25)
                q75_value = np.percentile(values, 75)

                col_stats_dict['Mean'] = round(mean_value, 3)
                col_stats_dict['Median'] = round(median_value, 3)
                col_stats_dict['Std Dev'] = round(std_value, 3)
                col_stats_dict['Mean ± Std Dev'] = f"{round(mean_value, 2)} \u00B1 {round(std_value, 2)}"
                col_stats_dict['Variance'] = round(var_value, 3)

                min_display = 'NA'
                if not pd.isna(min_value):
                    min_display = int(min_value) if min_value == int(
                        min_value) else round(min_value, 3)
                col_stats_dict['Min'] = min_display

                max_display = 'NA'
                if not pd.isna(max_value):
                    max_display = int(max_value) if max_value == int(
                        max_value) else round(max_value, 3)
                col_stats_dict['Max'] = max_display

                col_stats_dict['25%'] = round(q25_value, 3)
                col_stats_dict['50%'] = round(median_value, 3)
                col_stats_dict['75%'] = round(q75_value, 3)

                col_stats_dict['Skewness'] = round(
                    col_data.skew(), 3) if col_stats_dict['Count'] > 2 else 'NA'
                col_stats_dict['Kurtosis'] = round(
                    col_data.kurtosis(), 3) if col_stats_dict['Count'] > 3 else 'NA'

                mode_series = col_data.mode()
                if not mode_series.empty:
                    mode_val = mode_series.iloc[0]
                    col_stats_dict['Mode'] = round(
                        mode_val, 3) if pd.api.types.is_number(mode_val) else mode_val
                else:
                    col_stats_dict['Mode'] = 'NA'
        else:
            if col_stats_dict['Count'] > 0:
                col_stats_dict['Unique Values'] = col_data.nunique()
                mode_series = col_data.mode()
                if not mode_series.empty:
                    mode_val = mode_series.iloc[0]
                    col_stats_dict['Mode'] = mode_val

                    value_counts_series = col_data.value_counts()
                    mode_freq = value_counts_series.get(mode_val, 0)

                    col_stats_dict['Mode Frequency'] = mode_freq
                    col_stats_dict['Mode Percentage'] = round(
                        (mode_freq / col_stats_dict['Count']) * 100, 2) if col_stats_dict['Count'] > 0 else 'NA'

        stats[col] = col_stats_dict

    stats_df = pd.DataFrame(stats)
    final_stats_df = stats_df.fillna('NA')

    # Per your original file, converting all to string for final output.
    # If Mean ± Std Dev is already a string, the astype(str) for that column is fine.
    # The explicit loop you had for this is not strictly needed if the whole df is astype(str).
    return final_stats_df.astype(str)

=======
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                values = df[col].dropna().values
                if df[col].count() == 0:
                    col_stats = {'Count': 0, 'Missing': df[col].isnull().sum(), 'Mean': 'NA', 'Median': 'NA', 'Mode': 'NA', 'Std Dev': 'NA', 'Mean \u00B1 Std Dev': 'NA',
                                 'Variance': 'NA', 'Min': 'NA', 'Max': 'NA', '25%': 'NA', '50%': 'NA', '75%': 'NA', 'Skewness': 'NA', 'Kurtosis': 'NA'}
                else:
                    mean_value = values.mean() if len(values) > 0 else np.nan
                    std_value = values.std() if len(values) > 0 else np.nan
                    min_value = values.min() if len(values) > 0 else np.nan
                    max_value = values.max() if len(values) > 0 else np.nan

                    if np.isnan(mean_value) or np.isnan(std_value):
                        mean_std_format, mean_value_display, std_value_display = 'NA', 'NA', 'NA'
                    else:
                        mean_formatted, std_formatted = f"{mean_value:.2f}", f"{std_value:.2f}"
                        mean_std_format = f"{mean_formatted} \u00B1 {std_formatted}"
                        mean_value_display, std_value_display = round(
                            mean_value, 3), round(std_value, 3)

                    mode_value = df[col].mode(
                    ).iloc[0] if not df[col].mode().empty else 'NA'
                    if not isinstance(mode_value, str) and not np.isnan(mode_value):
                        mode_value = round(mode_value, 3)

                    median = np.median(values) if len(values) > 0 else np.nan
                    median_display = round(
                        median, 3) if not np.isnan(median) else 'NA'
                    variance = np.var(values) if len(values) > 0 else np.nan
                    variance_display = round(
                        variance, 3) if not np.isnan(variance) else 'NA'
                    min_val = np.min(values) if len(values) > 0 else np.nan
                    max_val = np.max(values) if len(values) > 0 else np.nan
                    q25 = np.percentile(values, 25) if len(
                        values) > 0 else np.nan
                    q25_display = round(q25, 3) if not np.isnan(q25) else 'NA'
                    q50 = np.percentile(values, 50) if len(
                        values) > 0 else np.nan
                    q50_display = round(q50, 3) if not np.isnan(q50) else 'NA'
                    q75 = np.percentile(values, 75) if len(
                        values) > 0 else np.nan
                    q75_display = round(q75, 3) if not np.isnan(q75) else 'NA'
                    skew = pd.Series(values).skew() if len(
                        values) > 2 else np.nan
                    skew_display = round(
                        skew, 3) if not np.isnan(skew) else 'NA'
                    kurt = pd.Series(values).kurtosis() if len(
                        values) > 3 else np.nan
                    kurt_display = round(
                        kurt, 3) if not np.isnan(kurt) else 'NA'

                    min_display = int(min_val) if isinstance(min_val, (int, np.integer)) and not pd.isna(min_val) else round(
                        min_val, 3) if isinstance(min_val, (float, np.floating)) and not pd.isna(min_val) else 'NA'
                    max_display = int(max_val) if isinstance(max_val, (int, np.integer)) and not pd.isna(max_val) else round(
                        max_val, 3) if isinstance(max_val, (float, np.floating)) and not pd.isna(max_val) else 'NA'

                    col_stats = {'Count': df[col].count(), 'Missing': df[col].isnull().sum(), 'Mean': mean_value_display, 'Median': median_display, 'Mode': mode_value, 'Std Dev': std_value_display, 'Mean \u00B1 Std Dev': mean_std_format,
                                 'Variance': variance_display, 'Min': min_display, 'Max': max_display, '25%': q25_display, '50%': q50_display, '75%': q75_display, 'Skewness': skew_display, 'Kurtosis': kurt_display}
            except Exception as e:
                col_stats = {'Count': df[col].count(), 'Missing': df[col].isnull().sum(), 'Mean': 'NA', 'Median': 'NA', 'Mode': 'NA', 'Std Dev': 'NA',
                             'Mean \u00B1 Std Dev': 'NA', 'Variance': 'NA', 'Min': 'NA', 'Max': 'NA', '25%': 'NA', '50%': 'NA', '75%': 'NA', 'Skewness': 'NA', 'Kurtosis': 'NA'}
            stats[col] = col_stats
        else:
            try:
                n_unique = df[col].nunique()
                if df[col].count() == 0 or df[col].value_counts().empty:
                    mode_value, mode_freq, mode_pct_rounded = 'NA', 'NA', 'NA'
                else:
                    mode_value = df[col].mode(
                    ).iloc[0] if not df[col].mode().empty else 'NA'
                    mode_freq = df[col].value_counts(
                    ).iloc[0] if not df[col].value_counts().empty else 'NA'
                    mode_pct = (mode_freq / df[col].count() * 100) if isinstance(
                        mode_freq, (int, float)) and df[col].count() > 0 else 'NA'
                    mode_pct_rounded = round(mode_pct, 2) if isinstance(
                        mode_pct, (int, float)) else 'NA'
                col_stats = {'Count': df[col].count(), 'Missing': df[col].isnull().sum(), 'Unique Values': n_unique, 'Mode': mode_value, 'Mode Frequency': mode_freq, 'Mode Percentage': mode_pct_rounded, 'Mean': 'NA',
                             'Median': 'NA', 'Std Dev': 'NA', 'Mean \u00B1 Std Dev': 'NA', 'Variance': 'NA', 'Min': 'NA', 'Max': 'NA', '25%': 'NA', '50%': 'NA', '75%': 'NA', 'Skewness': 'NA', 'Kurtosis': 'NA'}
            except Exception as e:
                col_stats = {'Count': df[col].count(), 'Missing': df[col].isnull().sum(), 'Mean': 'NA', 'Median': 'NA', 'Mode': 'NA', 'Std Dev': 'NA', 'Mean \u00B1 Std Dev': 'NA', 'Variance': 'NA',
                             'Min': 'NA', 'Max': 'NA', '25%': 'NA', '50%': 'NA', '75%': 'NA', 'Skewness': 'NA', 'Kurtosis': 'NA', 'Unique Values': 'NA', 'Mode Frequency': 'NA', 'Mode Percentage': 'NA'}
            stats[col] = col_stats

    stats_df = pd.DataFrame(stats)
    for col_name in stats_df.columns:
        if 'Mean \u00B1 Std Dev' in stats_df.index and isinstance(stats_df.loc['Mean \u00B1 Std Dev', col_name], str):
            stats_df[col_name] = stats_df[col_name].astype(str)
        for idx_name in ['Mean', 'Median', 'Mode', 'Std Dev', 'Variance', 'Min', 'Max', '25%', '50%', '75%', 'Skewness', 'Kurtosis', 'Mode Percentage']:
            if idx_name in stats_df.index and not isinstance(stats_df.loc[idx_name, col_name], (int, float, np.number)) and not pd.isna(stats_df.loc[idx_name, col_name]) and stats_df.loc[idx_name, col_name] != 'NA':
                stats_df.loc[idx_name, col_name] = str(
                    stats_df.loc[idx_name, col_name])
    return stats_df
>>>>>>> Stashed changes


def infer_data_types(df):
<<<<<<< Updated upstream
    """
    Infer data types for each column in the dataframe
    """
    data_types = {}

    for col in df.columns:
        col_series = df[col]  # Use col_series for clarity
        if pd.api.types.is_numeric_dtype(col_series):
            if col_series.count() > 0:  # Only infer if there are non-NA values
                try:
                    # Check if all non-NA numeric values are whole numbers
                    is_whole_number = col_series.dropna().apply(
                        lambda x: float(x).is_integer()).all()
                    if is_whole_number:
                        # Check if it's likely categorical (low cardinality)
                        if col_series.nunique() < 10 or (col_series.nunique() / col_series.count()) < 0.05:  # Use count of non-NA for ratio
                            # Numeric but behaves like category
                            data_types[col] = 'categorical_numeric'
                        else:
                            data_types[col] = 'integer'
                    else:
                        data_types[col] = 'continuous'
                except Exception:  # If any conversion error occurs during lambda apply
                    # Default for problematic numeric
                    data_types[col] = 'continuous'
            else:  # All NA for a numeric dtype column
                # Default numeric type if all NaNs
                data_types[col] = 'continuous'
        # handles timezone-aware datetimes too
        elif pd.api.types.is_datetime64_any_dtype(col_series):
            data_types[col] = 'datetime'
        elif pd.api.types.is_categorical_dtype(col_series):
            data_types[col] = 'categorical'
        else:  # Object Dtypes
            # Attempt to convert to datetime if a good portion can be converted
            # This helps identify object columns that are actually datetimes
            datetime_converted_count = 0
            if col_series.count() > 0:  # Only attempt if there are non-NA values
                try:
                    datetime_converted_count = pd.to_datetime(
                        col_series, errors='coerce', infer_datetime_format=True).count()
                # Catch broad errors during conversion attempt
                except (ValueError, TypeError, OverflowError):
                    pass  # Keep datetime_converted_count as 0

            # If >80% are convertible
            if datetime_converted_count > 0 and (datetime_converted_count / col_series.count()) > 0.8:
                data_types[col] = 'datetime'
            else:  # Fallback to text/categorical based on cardinality for object type
                if col_series.nunique() < 10 or (col_series.count() > 0 and (col_series.nunique() / col_series.count()) < 0.05):
                    data_types[col] = 'categorical'
                else:
                    data_types[col] = 'text'
=======
    data_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].dropna().apply(lambda x: x == int(x) if pd.notnull(x) and isinstance(x, (int, float)) and x == x else False).all():
                if df[col].nunique() / len(df) < 0.05 or df[col].nunique() < 10:
                    data_types[col] = 'categorical_numeric'
                else:
                    data_types[col] = 'integer'
            else:
                data_types[col] = 'continuous'
        elif pd.api.types.is_datetime64_dtype(df[col]):
            data_types[col] = 'datetime'
        else:
            if df[col].nunique() / len(df) < 0.05 or df[col].nunique() < 10:
                data_types[col] = 'categorical_object'
            else:
                data_types[col] = 'text'
>>>>>>> Stashed changes
    return data_types


def generate_age_groups(df, age_column, age_type='years', age_group_col_name='Generated Age Group'):
<<<<<<< Updated upstream
    """
    Generate age groups based on the age data
    """
    if age_column not in df.columns:
        raise ValueError(f"Age column '{age_column}' not found in DataFrame.")

    df_with_groups = df.copy()  # Work on a copy

    # Attempt to convert age column to numeric, coercing errors to NaN
    df_with_groups[age_column] = pd.to_numeric(
        df_with_groups[age_column], errors='coerce')

    valid_ages = df_with_groups[age_column].dropna()
    if valid_ages.empty:
        # If no valid ages, return original df and indicate no groups made
        # Use pandas NA for consistency
        df_with_groups[age_group_col_name] = pd.NA
        grouping_info = {'age_type': age_type, 'bin_size_description': 'NA',
                         'min_age_processed': 'NA', 'max_age_processed': 'NA', 'age_groups_generated': []}
        return df_with_groups, grouping_info, age_group_col_name

    min_age = valid_ages.min()
    max_age = valid_ages.max()
    # print(f"Unique valid ages: {sorted(valid_ages.unique())}, Min: {min_age}, Max: {max_age}")

    final_col_name = age_group_col_name
    # Avoid modifying the original age column if age_group_col_name is the same
=======
    if age_column not in df.columns:
        raise ValueError(f"Age column '{age_column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[age_column]):
        try:
            df[age_column] = pd.to_numeric(df[age_column])
        except ValueError:
            raise ValueError(
                f"Age column '{age_column}' is not numeric and could not be converted.")

    df_with_groups = df.copy()
    unique_ages = sorted(df_with_groups[age_column].dropna().unique())
    if not unique_ages:
        df_with_groups[age_group_col_name] = "Invalid Age Data"
        return df_with_groups, {'age_type': age_type, 'bin_size': 0, 'min_age': 'NA', 'max_age': 'NA', 'age_groups': []}, age_group_col_name

    min_age, max_age = min(unique_ages), max(unique_ages)
    final_col_name = age_group_col_name
>>>>>>> Stashed changes
    if age_group_col_name in df_with_groups.columns and age_group_col_name != age_column:
        counter = 1
        while f"{age_group_col_name}_{counter}" in df_with_groups.columns:
            counter += 1
        final_col_name = f"{age_group_col_name}_{counter}"

    def assign_age_group(age):
<<<<<<< Updated upstream
        if pd.isna(age):  # Handle NaN ages
            return pd.NA

        try:  # Ensure age is float for comparisons
            age = float(age)
        except ValueError:
            return pd.NA  # If age cannot be converted to float

        if age_type == 'years':
            if age < 18:
                start = int(age)  # Single year bins for < 18
                return f"{start}-{start} yrs"
            else:
                start = int(age // 5) * 5
                end = start + 4  # 5-year groups like 20-24, 25-29
                return f"{start}-{end} yrs"
        elif age_type == 'months':
            if age <= 24:
                start = int(age)  # Single month bins
                return f"{start}-{start} mo"
            else:
                start = int(age // 3) * 3
                end = start + 2  # 3-month groups
                return f"{start}-{end} mo"
        elif age_type == 'days':
            if age <= 90:  # Approx 3 months
                start = int(age // 7) * 7
                end = start + 6  # 7-day groups (1 week)
                return f"{start}-{end} days"
            else:
                weeks = int(age // 7)
                return f"{weeks}-{weeks} wk"  # Single week bins

        return f"Age: {age}"  # Fallback

    df_with_groups[final_col_name] = df_with_groups[age_column].apply(
        assign_age_group)

    generated_age_groups = sorted(
        list(df_with_groups[final_col_name].dropna().unique()))
    # print(f"Generated age groups: {generated_age_groups}")

    # Insert the age group column right after the age column if possible and distinct
    if age_column in df_with_groups.columns and final_col_name != age_column and final_col_name in df_with_groups.columns:
        try:
            original_columns = list(df_with_groups.columns)
            # The new column (final_col_name) is already in df_with_groups from the .apply()
            # We need to remove it first if we are reordering based on original_columns list
            if final_col_name in original_columns:
                original_columns.remove(final_col_name)

            age_col_idx = original_columns.index(age_column)

            new_col_order = original_columns[:age_col_idx+1] + \
                [final_col_name] + original_columns[age_col_idx+1:]
            df_with_groups = df_with_groups[new_col_order]
        # If age_column was not in original_columns (should not happen)
        except ValueError:
            pass
        except Exception as e:
            print(
                f"Warning: Could not reorder columns after age grouping: {e}")

    bin_size_info = 'Variable'
    if age_type == 'years':
        bin_size_info = '1 yr (<18), 5 yrs (>=18)'
    elif age_type == 'months':
        bin_size_info = '1 mo (<=24 mo), 3 mo (>24 mo)'
    elif age_type == 'days':
        bin_size_info = '7 days (<=90 days), 1 wk (>90 days)'

    grouping_info = {
        'age_type': age_type,
        'bin_size_description': bin_size_info,
        'min_age_processed': min_age if not valid_ages.empty else 'NA',
        'max_age_processed': max_age if not valid_ages.empty else 'NA',
        'age_groups_generated': generated_age_groups
    }
=======
        if pd.isna(age):
            return "Unknown Age"
        if age_type == 'years':
            if age < 1:
                return "0-0 yrs (Infant)"
            if age < 18:
                return f"{int(age // 1)}-{int(age // 1)} yrs"
            elif age < 70:
                return f"{int(age // 5) * 5}-{int(age // 5) * 5 + 4} yrs"
            else:
                return "70+ yrs"
        elif age_type == 'months':
            if age <= 24:
                return f"{int(age // 1)}-{int(age // 1)} mo"
            else:
                return f"{int(age // 3) * 3}-{int(age // 3) * 3 + 2} mo"
        elif age_type == 'days':
            if age <= 90:
                return f"{int(age // 7) * 7}-{int(age // 7) * 7 + 6} days"
            else:
                return f"{int(age // 7)}-{int(age // 7)} wk"
        return f"Age: {age}"

    df_with_groups[final_col_name] = df_with_groups[age_column].apply(
        assign_age_group)
    if age_column in df_with_groups.columns and final_col_name in df_with_groups.columns and age_column != final_col_name:
        age_col_idx = df_with_groups.columns.get_loc(age_column)
        cols = df_with_groups.columns.tolist()
        cols.remove(final_col_name)
        cols.insert(age_col_idx + 1, final_col_name)
        df_with_groups = df_with_groups[cols]
    generated_age_groups_sorted = sorted(
        list(df_with_groups[final_col_name].dropna().unique()))
    grouping_info = {'age_type': age_type, 'min_age': min_age,
                     'max_age': max_age, 'age_groups': generated_age_groups_sorted}
>>>>>>> Stashed changes
    return df_with_groups, grouping_info, final_col_name


def analyze_master_parameters(df, master_columns, analysis_columns=None):
    if not master_columns:
<<<<<<< Updated upstream
        return {}

    if analysis_columns is None:
        analysis_columns = [
            col for col in df.columns if col not in master_columns]

    results_by_master_col = {}

    for master_col in master_columns:
        if master_col not in df.columns:
            results_by_master_col[master_col] = {
                'type': 'error', 'error': f"Master column '{master_col}' not found."}
            continue

        current_master_results = {'type': 'categorical', 'groups': {}}
        master_col_series = df[master_col]

        if pd.api.types.is_numeric_dtype(master_col_series):
            current_master_results['type'] = 'numerical'
            # For numerical master, group by unique values if cardinality is low, else bin
            # Ensure there's enough data for meaningful binning
            if master_col_series.nunique() > 10 and master_col_series.count() > (master_col_series.nunique() * 2):
                try:
                    # Bins based on unique non-NA values
                    num_unique_for_bins = master_col_series.dropna().nunique()
                    num_bins = min(
                        5, num_unique_for_bins) if num_unique_for_bins > 1 else 1
                    if num_bins > 1:
                        binned_master_col = pd.cut(
                            master_col_series, bins=num_bins, precision=3, duplicates='drop')
                        grouped = df.groupby(binned_master_col, observed=False)
                    else:  # Not enough unique values to make more than 1 bin
                        # Group by actual values
                        grouped = df.groupby(master_col_series)
                except Exception as e_bin:
                    # print(f"Binning failed for {master_col}: {e_bin}. Grouping by unique values.")
                    # Fallback to unique values
                    grouped = df.groupby(master_col_series)
            else:  # Low cardinality numeric or not enough data to bin well
                grouped = df.groupby(master_col_series)
        else:  # Categorical master
            grouped = df.groupby(master_col_series)

        for group_name, group_df in grouped:
            # Ensure group name is string for keys
            str_group_name = str(group_name)

            # Descriptive stats are calculated on a copy of the group_df
            group_des_stats = get_descriptive_stats(
                group_df.copy(), analysis_columns)

            current_master_results['groups'][str_group_name] = {
                'count': len(group_df),
                'stats': group_des_stats
            }
        results_by_master_col[master_col] = current_master_results

    return results_by_master_col

=======
        return None
    if analysis_columns is None:
        analysis_columns = [
            col for col in df.columns if col not in master_columns]
    results = {}
    for master_col in master_columns:
        try:
            if master_col not in df.columns:
                continue
            col_type = "categorical"
            if pd.api.types.is_numeric_dtype(df[master_col]):
                col_type = "numerical_continuous" if df[master_col].nunique(
                ) > 10 else "numerical_discrete"
            col_results = {'type': col_type, 'groups': {}}
            if col_type == "categorical" or col_type == "numerical_discrete":
                groups = df[master_col].dropna().unique()
                try:
                    groups = sorted(groups, key=lambda x: (
                        isinstance(x, str), x))
                except TypeError:
                    groups = sorted(groups, key=str)
                for group_val in groups:
                    group_df = df[df[master_col] == group_val]
                    group_stats_df = get_descriptive_stats(
                        group_df, analysis_columns)
                    col_results['groups'][str(group_val)] = {'count': len(
                        group_df), 'stats': group_stats_df, 'data': group_df}
            elif col_type == "numerical_continuous":
                try:
                    bins = pd.qcut(df[master_col], q=5, duplicates='drop')
                except ValueError:
                    bins = pd.cut(df[master_col], bins=5, duplicates='drop')
                for bin_interval in bins.cat.categories:
                    group_df = df[bins == bin_interval]
                    group_stats_df = get_descriptive_stats(
                        group_df, analysis_columns)
                    col_results['groups'][str(bin_interval)] = {'count': len(
                        group_df), 'stats': group_stats_df, 'data': group_df}
            results[master_col] = col_results
        except Exception as e:
            results[master_col] = {'type': 'error', 'error': str(e)}
    return results
>>>>>>> Stashed changes


def detect_outliers(df, column, method='iqr'):
    if column not in df.columns:
<<<<<<< Updated upstream
        return pd.Series([False] * len(df), index=df.index), {'method': method, 'num_outliers': 0,
                                                              'percentage_outliers': 0.0, 'error': f"Column '{column}' not found."}

    col_data = df[column]
    if not pd.api.types.is_numeric_dtype(col_data):
        return pd.Series([False] * len(df), index=df.index), {'method': method, 'num_outliers': 0,
                                                              'percentage_outliers': 0.0, 'error': 'Column is not numeric.'}

    # Initialize with all False
    outliers_mask = pd.Series([False] * len(df), index=df.index)
    stats = {'method': method, 'num_outliers': 0, 'percentage_outliers': 0.0}

    col_data_dropna = col_data.dropna()
    if col_data_dropna.empty:  # If no actual data points after dropping NA, return
        return outliers_mask, stats

    if method == 'iqr':
        Q1 = col_data_dropna.quantile(0.25)
        Q3 = col_data_dropna.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers in the non-NA data
        outliers_condition = (col_data_dropna < lower_bound) | (
            col_data_dropna > upper_bound)
        # Map these back to the original DataFrame's index using .loc
        outliers_mask.loc[col_data_dropna[outliers_condition].index] = True

        stats.update({'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                     'lower_bound': lower_bound, 'upper_bound': upper_bound})

    elif method == 'zscore':
        mean_val = col_data_dropna.mean()
        std_val = col_data_dropna.std()

        # Avoid division by zero or NaN std
        if std_val == 0 or pd.isna(std_val):
            return outliers_mask, stats  # No outliers if std is 0 or NaN

        z_scores = (col_data_dropna - mean_val) / std_val
        # Standard threshold for z-score
        outliers_condition = (z_scores.abs() > 3)
        outliers_mask.loc[col_data_dropna[outliers_condition].index] = True

        stats.update({'mean': mean_val, 'std': std_val})
    else:
        raise ValueError(
            f"Unknown outlier detection method: {method}. Choose 'iqr' or 'zscore'.")

    # Ensure num_outliers is an int
    stats['num_outliers'] = int(outliers_mask.sum())
    stats['percentage_outliers'] = round(
        (stats['num_outliers'] / len(df) * 100), 2) if len(df) > 0 else 0.0

=======
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is not numeric, cannot detect outliers.")
    data_series = df[column].dropna()
    if data_series.empty:
        return pd.Series([False]*len(df), index=df.index), {'method': method, 'num_outliers': 0, 'percentage_outliers': 0}
    outliers_mask = pd.Series([False]*len(df), index=df.index)
    stats = {}
    if method == 'iqr':
        Q1, Q3 = data_series.quantile(0.25), data_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        series_outliers = (data_series < lower_bound) | (
            data_series > upper_bound)
        outliers_mask.loc[data_series[series_outliers].index] = True
        stats = {'method': 'IQR', 'Q1': Q1, 'Q3': Q3, 'IQR': IQR, 'lower_bound': lower_bound, 'upper_bound': upper_bound,
                 'num_outliers': outliers_mask.sum(), 'percentage_outliers': (outliers_mask.sum() / len(df) * 100) if len(df) > 0 else 0}
    elif method == 'zscore':
        mean, std = data_series.mean(), data_series.std()
        if std == 0:
            series_outliers = pd.Series(
                [False]*len(data_series), index=data_series.index)
        else:
            z_scores = (data_series - mean) / std
            series_outliers = (z_scores.abs() > 3)
        outliers_mask.loc[data_series[series_outliers].index] = True
        stats = {'method': 'Z-score', 'mean': mean, 'std': std, 'num_outliers': outliers_mask.sum(
        ), 'percentage_outliers': (outliers_mask.sum() / len(df) * 100) if len(df) > 0 else 0}
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
>>>>>>> Stashed changes
    return outliers_mask, stats


def calculate_manual_paired_differences(df, manual_pairs):
    df_enhanced = df.copy()
<<<<<<< Updated upstream
    pairs_info = {
        'pairs_processed': [],
        'difference_columns': [],  # Using 'difference_columns' as per your app.py expectation
        'summary': {}
    }

    for pre_col, post_col in manual_pairs:
        if pre_col in df_enhanced.columns and post_col in df_enhanced.columns:
            # Ensure both columns are numeric before attempting arithmetic
            if pd.api.types.is_numeric_dtype(df_enhanced[pre_col]) and pd.api.types.is_numeric_dtype(df_enhanced[post_col]):
                difference = df_enhanced[post_col] - df_enhanced[pre_col]

                # Try to derive a base name more robustly
                base_name_pre = re.sub(
                    r'(?i)^(pre[\s_-]?|before[\s_-]?|baseline[\s_-]?|initial[\s_-]?)', '', pre_col, count=1).strip()
                base_name_post = re.sub(
                    r'(?i)^(post[\s_-]?|after[\s_-]?|final[\s_-]?|end[\s_-]?)', '', post_col, count=1).strip()

                base_name = base_name_pre if base_name_pre and base_name_pre.lower(
                ) == base_name_post.lower() else f"{pre_col}_to_{post_col}"
                # Fallback if base_name_pre itself was empty after stripping prefixes
                if not base_name_pre and base_name_post:  # If pre is empty but post is not, use post
                    base_name = base_name_post
                elif not base_name_pre and not base_name_post:  # If both are empty, use column names
                    base_name = f"{pre_col}_to_{post_col}"

                diff_col_name_candidate = f"{base_name} Difference"

                actual_diff_col_name = diff_col_name_candidate
                counter = 1
                while actual_diff_col_name in df_enhanced.columns:
                    actual_diff_col_name = f"{diff_col_name_candidate}_{counter}"
                    counter += 1

                df_enhanced[actual_diff_col_name] = difference

                pairs_info['pairs_processed'].append({
                    'pre_column': pre_col,
                    'post_column': post_col,
                    'difference_column': actual_diff_col_name,
                    'base_parameter_name_used': base_name,
                    'mean_difference': round(difference.mean(), 3) if not pd.isna(difference.mean()) else 'NA',
                    'std_difference': round(difference.std(), 3) if not pd.isna(difference.std()) else 'NA',
                    'min_difference': round(difference.min(), 3) if not pd.isna(difference.min()) else 'NA',
                    'max_difference': round(difference.max(), 3) if not pd.isna(difference.max()) else 'NA'
                })
                pairs_info['difference_columns'].append(actual_diff_col_name)
            else:
                print(
                    f"Warning: Pair ({pre_col}, {post_col}) not both numeric. Skipping difference calculation.")
        else:
            print(
                f"Warning: One or both columns for pair ({pre_col}, {post_col}) not found in DataFrame. Skipping.")

    pairs_info['summary'] = {
        'total_manual_pairs_input': len(manual_pairs),
        'total_pairs_processed_successfully': len(pairs_info['pairs_processed']),
        'total_difference_columns_added': len(pairs_info['difference_columns']),
    }
    return df_enhanced, pairs_info


def calculate_paired_differences(df):  # Marked as DEPRECATED by user
    """
    DEPRECATED: This function is kept for backward compatibility but should not be used.
    Use calculate_manual_paired_differences instead for manual pair selection.
    """
    df_enhanced = df.copy()
    pairs_info = {
        'pairs_found': [],
        'difference_columns': [],
        'summary': {}
    }
    columns = df.columns.tolist()
    # Added (?i) for case-insensitivity to regex patterns
    pre_patterns = [r'(?i)pre[\s_-]?', r'(?i)before[\s_-]?',
                    r'(?i)baseline[\s_-]?', r'(?i)initial[\s_-]?']
    post_patterns = [r'(?i)post[\s_-]?', r'(?i)after[\s_-]?',
                     r'(?i)final[\s_-]?', r'(?i)end[\s_-]?']
    potential_pairs = []
    processed_cols_as_pre = set()

    for col in columns:
        if col in processed_cols_as_pre:
            continue

        base_name_from_pre = ""
        is_pre_col = False
        for pre_pattern in pre_patterns:
            # Using match for start of string
            match_pre = re.match(pre_pattern, col)
            if match_pre:
                # Extract part after the prefix
                base_name_from_pre = col[match_pre.end():].strip()
                # Clean leading/trailing separators from base_name
                base_name_from_pre = re.sub(
                    r'^[\s_-]+|[\s_-]+$', '', base_name_from_pre)
                is_pre_col = True
                # Mark as processed to avoid re-evaluating as pre
                processed_cols_as_pre.add(col)
                break  # Found a pre-pattern, no need to check others for this column

        if is_pre_col:
            for post_col in columns:
                if col == post_col:
                    continue  # Don't pair a column with itself

                for post_pattern in post_patterns:
                    # Using match for start of string
                    match_post = re.match(post_pattern, post_col)
                    if match_post:
                        base_name_from_post = post_col[match_post.end(
                        ):].strip()
                        base_name_from_post = re.sub(
                            r'^[\s_-]+|[\s_-]+$', '', base_name_from_post)

                        # Compare base names case-insensitively, ensure not empty
                        if base_name_from_pre and base_name_from_pre.lower() == base_name_from_post.lower():
                            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[post_col]):
                                # Avoid adding duplicate pairs (e.g. if (A,B) is found, don't add (B,A) later if logic implies direction)
                                # Simple check
                                if not any(p[0] == post_col and p[1] == col for p in potential_pairs):
                                    # Store original pre_col's base name
                                    potential_pairs.append(
                                        (col, post_col, base_name_from_pre))
                            break  # Matched post_pattern for this post_col, move to next post_col or pre_col

    # Track (pre,post) tuples to avoid redundant diff calculations
    unique_pairs_processed_for_diff = []

    for pre_col, post_col, base_name_for_diff in potential_pairs:
        if (pre_col, post_col) in unique_pairs_processed_for_diff:
            # Skip if this pair was already processed (e.g. from multiple pattern matches leading to same pair)
            continue

        try:
            difference = df_enhanced[post_col] - df_enhanced[pre_col]

            # Use the base_name that was captured when the pair was identified
            display_base_name = base_name_for_diff if base_name_for_diff else pre_col.replace(
                "Pre", "").strip()  # Fallback

            if display_base_name:  # Ensure base name is not empty for title case
                diff_col_name_candidate = f"{display_base_name.title()} Difference"
            else:
                diff_col_name_candidate = f"Difference ({post_col} - {pre_col})"

            actual_diff_col_name = diff_col_name_candidate
            counter = 1
            while actual_diff_col_name in df_enhanced.columns:
                actual_diff_col_name = f"{diff_col_name_candidate}_{counter}"
                counter += 1

            df_enhanced[actual_diff_col_name] = difference

            pairs_info['pairs_found'].append({
                'pre_column': pre_col,
                'post_column': post_col,
                'difference_column': actual_diff_col_name,
                'base_parameter': display_base_name,
                'mean_difference': round(difference.mean(), 3) if not pd.isna(difference.mean()) else 'NA',
                'std_difference': round(difference.std(), 3) if not pd.isna(difference.std()) else 'NA',
                'min_difference': round(difference.min(), 3) if not pd.isna(difference.min()) else 'NA',
                'max_difference': round(difference.max(), 3) if not pd.isna(difference.max()) else 'NA'
            })
            pairs_info['difference_columns'].append(actual_diff_col_name)
            unique_pairs_processed_for_diff.append((pre_col, post_col))
=======
    pairs_info = {'pairs_processed': [],
                  'difference_columns': [], 'summary': {}}
    for pre_col, post_col in manual_pairs:
        try:
            if pre_col in df.columns and post_col in df.columns and pd.api.types.is_numeric_dtype(df[pre_col]) and pd.api.types.is_numeric_dtype(df[post_col]):
                difference = df_enhanced[post_col] - df_enhanced[pre_col]
                base_name = pre_col.replace('Pre ', '').replace('Post ', '').replace('pre ', '').replace(
                    'post ', '').replace('Pre', '').replace('Post', '').replace('pre', '').replace('post', '').strip()
                diff_col_name = f"{base_name} Difference" if base_name else f"{post_col} - {pre_col} Difference"
                original_diff_col_name = diff_col_name
                counter = 1
                while diff_col_name in df_enhanced.columns:
                    diff_col_name = f"{original_diff_col_name}_{counter}"
                    counter += 1
                df_enhanced[diff_col_name] = difference
                pairs_info['pairs_processed'].append({'pre_column': pre_col, 'post_column': post_col, 'difference_column': diff_col_name, 'base_parameter': base_name, 'mean_difference': difference.mean(
                ), 'std_difference': difference.std(), 'min_difference': difference.min(), 'max_difference': difference.max()})
                pairs_info['difference_columns'].append(diff_col_name)
        except Exception as e:
            continue
    pairs_info['summary'] = {'total_pairs_processed': len(pairs_info['pairs_processed']), 'total_difference_columns_added': len(
        pairs_info['difference_columns']), 'original_columns': len(df.columns), 'enhanced_columns': len(df_enhanced.columns)}
    return df_enhanced, pairs_info


def calculate_paired_differences(df):
    df_enhanced = df.copy()
    pairs_info = {'pairs_found': [], 'difference_columns': [], 'summary': {}}
    columns = df.columns.tolist()
    pre_patterns = [r'pre[\s_-]?', r'before[\s_-]?',
                    r'baseline[\s_-]?', r'initial[\s_-]?']
    post_patterns = [r'post[\s_-]?', r'after[\s_-]?',
                     r'final[\s_-]?', r'end[\s_-]?']
    potential_pairs = []
    for col in columns:
        col_lower = col.lower()
        for pre_pattern in pre_patterns:
            if re.search(pre_pattern, col_lower):
                base_name = re.sub(pre_pattern, '', col_lower).strip()
                base_name = re.sub(r'^[\s_-]+|[\s_-]+$', '', base_name)
                for post_col in columns:
                    post_col_lower = post_col.lower()
                    for post_pattern in post_patterns:
                        if re.search(post_pattern, post_col_lower):
                            post_base = re.sub(
                                post_pattern, '', post_col_lower).strip()
                            post_base = re.sub(
                                r'^[\s_-]+|[\s_-]+$', '', post_base)
                            if (base_name == post_base or (len(base_name) > 3 and base_name in post_base)) and pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[post_col]):
                                potential_pairs.append(
                                    (col, post_col, base_name))
    unique_pairs = []
    processed_bases = set()
    for pre_col, post_col, base_name in potential_pairs:
        pair_tuple = tuple(sorted((pre_col, post_col)))
        identifier = f"{pair_tuple}_{base_name[:5]}"
        if identifier not in processed_bases:
            unique_pairs.append((pre_col, post_col, base_name))
            processed_bases.add(identifier)
    for pre_col, post_col, base_name in unique_pairs:
        try:
            difference = df_enhanced[post_col] - df_enhanced[pre_col]
            diff_col_name = f"{base_name.title()} Difference" if base_name else f"Difference ({post_col} - {pre_col})"
            original_name = diff_col_name
            counter = 1
            while diff_col_name in df_enhanced.columns:
                diff_col_name = f"{original_name}_{counter}"
                counter += 1
            df_enhanced[diff_col_name] = difference
            pairs_info['pairs_found'].append({'pre_column': pre_col, 'post_column': post_col, 'difference_column': diff_col_name, 'base_parameter': base_name,
                                             'mean_difference': difference.mean(), 'std_difference': difference.std(), 'min_difference': difference.min(), 'max_difference': difference.max()})
            pairs_info['difference_columns'].append(diff_col_name)
>>>>>>> Stashed changes
        except Exception as e:
            # print(f"Error processing pair ({pre_col}, {post_col}) for diff calculation: {e}") # Optional for debugging
            continue
<<<<<<< Updated upstream

    pairs_info['summary'] = {
        'total_pairs_found_and_processed': len(pairs_info['pairs_found']),
        'total_difference_columns_added': len(pairs_info['difference_columns']),
        'original_columns_count': len(df.columns),
        'enhanced_columns_count': len(df_enhanced.columns)
    }
=======
    pairs_info['summary'] = {'total_pairs_found': len(pairs_info['pairs_found']), 'total_difference_columns_added': len(
        pairs_info['difference_columns']), 'original_columns': len(df.columns), 'enhanced_columns': len(df_enhanced.columns)}
>>>>>>> Stashed changes
    return df_enhanced, pairs_info
