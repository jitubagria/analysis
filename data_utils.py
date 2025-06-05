import pandas as pd
import numpy as np
import io
import re


def load_data(uploaded_file):
    """
    Load data from uploaded file (CSV, Excel)
    """
    filename = uploaded_file.name
    file_ext = filename.split('.')[-1].lower()

    if file_ext == 'csv':
        try:
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
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
        except Exception as e:
            raise Exception(f"Failed to load Excel file: {str(e)}")
    else:
        raise Exception(f"Unsupported file format: {file_ext}")

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

    for col in columns:
        if col not in df.columns:
            continue

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


def infer_data_types(df):
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
    return data_types


def generate_age_groups(df, age_column, age_type='years', age_group_col_name='Generated Age Group'):
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
    if age_group_col_name in df_with_groups.columns and age_group_col_name != age_column:
        counter = 1
        while f"{age_group_col_name}_{counter}" in df_with_groups.columns:
            counter += 1
        final_col_name = f"{age_group_col_name}_{counter}"

    def assign_age_group(age):
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
    return df_with_groups, grouping_info, final_col_name


def analyze_master_parameters(df, master_columns, analysis_columns=None):
    if not master_columns:
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


def detect_outliers(df, column, method='iqr'):
    if column not in df.columns:
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
    return outliers_mask, stats


def calculate_manual_paired_differences(df, manual_pairs):
    df_enhanced = df.copy()
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
        except Exception as e:
            continue
    pairs_info['summary'] = {'total_pairs_found': len(pairs_info['pairs_found']), 'total_difference_columns_added': len(
        pairs_info['difference_columns']), 'original_columns': len(df.columns), 'enhanced_columns': len(df_enhanced.columns)}
    return df_enhanced, pairs_info
