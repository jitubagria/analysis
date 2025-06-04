import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def get_correlation_strength(r):
    """Helper function to determine correlation strength"""
    r_abs = abs(r)
    if r_abs < 0.3:
        return "Weak"
    elif r_abs < 0.5:
        return "Moderate"
    else:
        return "Strong"

def perform_ttest(df, col, test_type='one-sample', col2=None, mu=0, 
                 group_col=None, group1=None, group2=None, 
                 equal_var=False, alpha=0.05):
    """
    Perform t-test on the data
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    col : str
        The column name to analyze
    test_type : str
        Type of t-test to perform ('one-sample', 'two-sample-cols', 'two-sample-groups')
    col2 : str
        Second column name for two-sample t-test (when test_type='two-sample-cols')
    mu : float
        Population mean for one-sample t-test
    group_col : str
        Column containing group information (when test_type='two-sample-groups')
    group1, group2 : Any
        Group values to compare (when test_type='two-sample-groups')
    equal_var : bool
        Whether to assume equal variances for two-sample t-test
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    result = {}
    
    if test_type == 'one-sample':
        # Perform one-sample t-test
        data = df[col].dropna()
        t_stat, p_value = stats.ttest_1samp(data, mu)
        
        # Calculate confidence interval
        se = stats.sem(data)
        df_val = len(data) - 1
        ci_lower, ci_upper = stats.t.interval(1-alpha, df_val, loc=data.mean(), scale=se)
        
        result = {
            'sample_mean': data.mean(),
            't_stat': t_stat,
            'p_value': p_value,
            'df': df_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    elif test_type == 'two-sample-cols':
        # Perform two-sample t-test between two columns
        data1 = df[col].dropna()
        data2 = df[col2].dropna()
        
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        # Degrees of freedom
        if equal_var:
            df_val = len(data1) + len(data2) - 2
        else:
            # Welch-Satterthwaite equation for degrees of freedom
            s1_squared = np.var(data1, ddof=1)
            s2_squared = np.var(data2, ddof=1)
            n1 = len(data1)
            n2 = len(data2)
            
            num = (s1_squared/n1 + s2_squared/n2)**2
            denom = (s1_squared/n1)**2/(n1-1) + (s2_squared/n2)**2/(n2-1)
            df_val = num/denom
        
        # Calculate confidence interval for the difference
        mean1 = data1.mean()
        mean2 = data2.mean()
        mean_diff = mean1 - mean2
        
        if equal_var:
            # Pooled standard error
            s_pooled = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                               (len(data2) - 1) * np.var(data2, ddof=1)) / 
                              (len(data1) + len(data2) - 2))
            se_diff = s_pooled * np.sqrt(1/len(data1) + 1/len(data2))
        else:
            # Welch's t-test
            se_diff = np.sqrt(np.var(data1, ddof=1)/len(data1) + 
                             np.var(data2, ddof=1)/len(data2))
        
        ci_lower = mean_diff - stats.t.ppf(1-alpha/2, df_val) * se_diff
        ci_upper = mean_diff + stats.t.ppf(1-alpha/2, df_val) * se_diff
        
        result = {
            'mean1': mean1,
            'mean2': mean2,
            'mean_diff': mean_diff,
            't_stat': t_stat,
            'p_value': p_value,
            'df': df_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    elif test_type == 'two-sample-groups':
        # Perform two-sample t-test between two groups within a column
        data1 = df[df[group_col] == group1][col].dropna()
        data2 = df[df[group_col] == group2][col].dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            raise ValueError(f"One or both groups have no valid data for column '{col}'")
        
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        # Degrees of freedom
        if equal_var:
            df_val = len(data1) + len(data2) - 2
        else:
            # Welch-Satterthwaite equation for degrees of freedom
            s1_squared = np.var(data1, ddof=1)
            s2_squared = np.var(data2, ddof=1)
            n1 = len(data1)
            n2 = len(data2)
            
            num = (s1_squared/n1 + s2_squared/n2)**2
            denom = (s1_squared/n1)**2/(n1-1) + (s2_squared/n2)**2/(n2-1)
            df_val = num/denom
        
        # Calculate confidence interval for the difference
        mean1 = data1.mean()
        mean2 = data2.mean()
        mean_diff = mean1 - mean2
        
        if equal_var:
            # Pooled standard error
            s_pooled = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                               (len(data2) - 1) * np.var(data2, ddof=1)) / 
                              (len(data1) + len(data2) - 2))
            se_diff = s_pooled * np.sqrt(1/len(data1) + 1/len(data2))
        else:
            # Welch's t-test
            se_diff = np.sqrt(np.var(data1, ddof=1)/len(data1) + 
                             np.var(data2, ddof=1)/len(data2))
        
        ci_lower = mean_diff - stats.t.ppf(1-alpha/2, df_val) * se_diff
        ci_upper = mean_diff + stats.t.ppf(1-alpha/2, df_val) * se_diff
        
        result = {
            'mean1': mean1,
            'mean2': mean2,
            'mean_diff': mean_diff,
            't_stat': t_stat,
            'p_value': p_value,
            'df': df_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    return result

def perform_anova(df, num_col, cat_col, alpha=0.05):
    """
    Perform one-way ANOVA
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    num_col : str
        Numerical column (dependent variable)
    cat_col : str
        Categorical column (factor)
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Dictionary containing ANOVA results
    """
    # Create a clean dataframe for analysis
    anova_df = df[[num_col, cat_col]].dropna()
    
    # Check if there's enough data
    if len(anova_df) == 0:
        raise ValueError("No valid data for ANOVA analysis")
    
    # Calculate group statistics
    group_stats = anova_df.groupby(cat_col)[num_col].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    # Add confidence intervals
    group_stats['sem'] = group_stats.apply(lambda x: x['std'] / np.sqrt(x['count']), axis=1)
    group_stats['ci_lower'] = group_stats.apply(
        lambda x: x['mean'] - stats.t.ppf(1-alpha/2, x['count']-1) * x['sem'], axis=1
    )
    group_stats['ci_upper'] = group_stats.apply(
        lambda x: x['mean'] + stats.t.ppf(1-alpha/2, x['count']-1) * x['sem'], axis=1
    )
    
    # Fit the ANOVA model using statsmodels
    formula = f"{num_col} ~ C({cat_col})"
    model = ols(formula, data=anova_df).fit()
    anova_table = anova_lm(model)
    
    # Extract F-statistic and p-value
    f_stat = anova_table.iloc[0, 2]
    p_value = anova_table.iloc[0, 3]
    
    # Format ANOVA table
    anova_table_formatted = anova_table.reset_index()
    anova_table_formatted.columns = ['Source', 'DF', 'Sum of Squares', 'Mean Square', 'F Value', 'Pr(>F)']
    
    result = {
        'f_stat': f_stat,
        'p_value': p_value,
        'anova_table': anova_table_formatted,
        'group_stats': group_stats
    }
    
    return result

def perform_correlation_analysis(df, columns, method='pearson', alpha=0.05):
    """
    Perform correlation analysis on selected columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    columns : list
        List of columns to analyze
    method : str
        Correlation method ('pearson', 'spearman', or 'kendall')
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Dictionary containing correlation results
    """
    # First filter to only include numeric columns for correlation analysis
    numeric_columns = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
    
    # If no numeric columns, return appropriate placeholders
    if len(numeric_columns) < 2:
        empty_df = pd.DataFrame(index=columns, columns=columns)
        return {
            'corr_matrix': empty_df,
            'p_matrix': empty_df.copy(),
            'significant_corrs': pd.DataFrame(columns=['Variable 1', 'Variable 2', 
                                                      f'{method.capitalize()} Correlation', 
                                                      'p-value', 'Significant'])
        }
    
    # Extract the selected numeric columns and drop missing values
    corr_df = df[numeric_columns].dropna()
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr(method=method)
    
    # Initialize p-value matrix
    p_matrix = pd.DataFrame(np.nan, index=corr_matrix.index, columns=corr_matrix.columns)
    
    # Calculate p-values for each correlation
    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            if i != j:  # Skip the diagonal
                try:
                    x = corr_df[numeric_columns[i]]
                    y = corr_df[numeric_columns[j]]
                    
                    if method == 'pearson':
                        _, p_value = stats.pearsonr(x, y)
                    elif method == 'spearman':
                        _, p_value = stats.spearmanr(x, y)
                    elif method == 'kendall':
                        _, p_value = stats.kendalltau(x, y)
                    
                    p_matrix.iloc[i, j] = p_value
                except Exception:
                    # If there's an error calculating correlation, set to NaN
                    p_matrix.iloc[i, j] = np.nan
    
    # Fill diagonal with 0 p-values
    for i in range(len(numeric_columns)):
        p_matrix.iloc[i, i] = 0
    
    # Find significant correlations
    significant_mask = (p_matrix < alpha) & (p_matrix != 0) & (~p_matrix.isna())  # Exclude diagonal and NaNs
    
    # Create a dataframe of significant correlations
    significant_pairs = []
    
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):  # Only upper triangle to avoid duplicates
            if significant_mask.iloc[i, j]:
                significant_pairs.append({
                    'Variable 1': numeric_columns[i],
                    'Variable 2': numeric_columns[j],
                    f'{method.capitalize()} Correlation': corr_matrix.iloc[i, j],
                    'p-value': p_matrix.iloc[i, j],
                    'Significant': 'Yes' if p_matrix.iloc[i, j] < alpha else 'No'
                })
    
    significant_corrs = pd.DataFrame(significant_pairs)
    
    # If the original columns included non-numeric ones, create a full matrix that includes all columns
    if len(numeric_columns) < len(columns):
        full_corr_matrix = pd.DataFrame(index=columns, columns=columns)
        full_p_matrix = pd.DataFrame(index=columns, columns=columns)
        
        # Fill in the values for numeric columns
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                full_corr_matrix.loc[col1, col2] = corr_matrix.iloc[i, j]
                full_p_matrix.loc[col1, col2] = p_matrix.iloc[i, j]
        
        corr_matrix = full_corr_matrix
        p_matrix = full_p_matrix
    
    # If no significant correlations, create an empty DataFrame with the right columns
    if significant_corrs.empty and not significant_pairs:
        significant_corrs = pd.DataFrame(columns=[
            'Variable 1', 'Variable 2', f'{method.capitalize()} Correlation', 
            'p-value', 'Significant'
        ])
    
    result = {
        'corr_matrix': corr_matrix,
        'p_matrix': p_matrix,
        'significant_corrs': significant_corrs
    }
    
    return result

def perform_chi_square_test(df, col1, col2, alpha=0.05):
    """
    Perform Chi-square test of independence on two categorical variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    col1 : str
        First categorical column
    col2 : str
        Second categorical column
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Dictionary containing Chi-square test results
    """
    # Drop rows with missing values in either column
    data = df[[col1, col2]].dropna()
    
    # Create contingency table
    contingency_table = pd.crosstab(data[col1], data[col2])
    
    # Perform Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Calculate Cramer's V (effect size)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
    
    # Determine significance and description
    significant = p_value < alpha
    
    if cramers_v < 0.1:
        effect_size = "Negligible"
    elif cramers_v < 0.3:
        effect_size = "Weak"
    elif cramers_v < 0.5:
        effect_size = "Moderate"
    else:
        effect_size = "Strong"
    
    # Calculate observed vs expected frequencies difference
    expected_df = pd.DataFrame(
        expected, 
        index=contingency_table.index,
        columns=contingency_table.columns
    )
    
    # Format contingency table for display
    formatted_table = contingency_table.copy()
    # Add row and column totals
    formatted_table['Total'] = formatted_table.sum(axis=1)
    formatted_table.loc['Total'] = formatted_table.sum()
    
    # Check if any expected frequencies are less than 5 (Chi-square assumption)
    expected_lt_5_count = (expected < 5).sum()
    expected_lt_5_pct = expected_lt_5_count / expected.size
    warning = None
    
    if expected_lt_5_pct > 0.2:
        warning = f"{expected_lt_5_count} cells ({expected_lt_5_pct:.1%}) have expected frequencies less than 5. Chi-square may not be valid."
    
    result = {
        'contingency_table': contingency_table,
        'formatted_table': formatted_table,
        'expected_frequencies': expected_df,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v,
        'effect_size': effect_size,
        'significant': significant,
        'n': n,
        'warning': warning
    }
    
    return result

def perform_pvalue_analysis(df, settings):
    """
    Perform comprehensive p-value analysis including overall and system-wise p-values
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    results = {
        'overall_pvalues': pd.DataFrame(),
        'system_pvalues': {}
    }
    
    columns = settings.get('columns', [])
    test_type = settings.get('test_type', 'Normality Test (Shapiro-Wilk)')
    alpha = settings.get('alpha', 0.05)
    master_columns = settings.get('master_columns', [])
    
    # Filter to only numeric columns for p-value analysis
    numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    # Overall p-value analysis
    overall_data = []
    
    for column in numeric_columns:
        try:
            data = df[column].dropna()
            if len(data) < 3:
                continue
                
            p_value = None
            test_statistic = None
            interpretation = ""
            
            if test_type == "Normality Test (Shapiro-Wilk)":
                if len(data) <= 5000:
                    test_stat, p_value = stats.shapiro(data)
                    test_statistic = test_stat
                    interpretation = "Normal distribution" if p_value > alpha else "Non-normal distribution"
                else:
                    test_stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                    test_statistic = test_stat
                    interpretation = "Normal distribution" if p_value > alpha else "Non-normal distribution"
                    
            elif test_type == "One-Sample T-Test":
                test_value = settings.get('test_value', 0.0)
                test_stat, p_value = stats.ttest_1samp(data, test_value)
                test_statistic = test_stat
                interpretation = f"Mean significantly different from {test_value}" if p_value < alpha else f"Mean not significantly different from {test_value}"
                
            elif test_type == "Two-Sample T-Test":
                group_column = settings.get('group_column')
                if group_column and group_column in df.columns:
                    groups = df[group_column].unique()
                    if len(groups) >= 2:
                        group1_data = df[df[group_column] == groups[0]][column].dropna()
                        group2_data = df[df[group_column] == groups[1]][column].dropna()
                        if len(group1_data) > 1 and len(group2_data) > 1:
                            test_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                            test_statistic = test_stat
                            interpretation = f"Significant difference between {groups[0]} and {groups[1]}" if p_value < alpha else f"No significant difference between {groups[0]} and {groups[1]}"
                            
            elif test_type == "ANOVA":
                group_column = settings.get('group_column')
                if group_column and group_column in df.columns:
                    groups = df[group_column].unique()
                    if len(groups) >= 2:
                        group_data = [df[df[group_column] == group][column].dropna() for group in groups]
                        group_data = [group for group in group_data if len(group) > 1]
                        if len(group_data) >= 2:
                            test_stat, p_value = stats.f_oneway(*group_data)
                            test_statistic = test_stat
                            interpretation = "Significant differences between groups" if p_value < alpha else "No significant differences between groups"
            
            if p_value is not None:
                # Calculate Mean ± SD for the parameter
                mean_val = data.mean()
                std_val = data.std()
                mean_sd = f"{mean_val:.2f} ± {std_val:.2f}"
                
                overall_data.append({
                    'Parameter': column,
                    'Mean ± SD': mean_sd,
                    'Test_Statistic': round(test_statistic, 4) if test_statistic else None,
                    'P_Value': round(p_value, 6),
                    'Significant': 'Yes' if p_value < alpha else 'No',
                    'Interpretation': interpretation
                })
                
        except Exception as e:
            continue
    
    results['overall_pvalues'] = pd.DataFrame(overall_data)
    
    # System-wise p-value analysis
    if master_columns:
        for master_col in master_columns:
            if master_col in df.columns:
                systems = df[master_col].unique()
                results['system_pvalues'][master_col] = {}
                
                for system in systems:
                    if pd.isna(system):
                        continue
                        
                    system_df = df[df[master_col] == system]
                    system_data = []
                    
                    for column in numeric_columns:
                        try:
                            data = system_df[column].dropna()
                            if len(data) < 3:
                                continue
                                
                            p_value = None
                            test_statistic = None
                            interpretation = ""
                            
                            if test_type == "Normality Test (Shapiro-Wilk)":
                                if len(data) <= 5000:
                                    test_stat, p_value = stats.shapiro(data)
                                    test_statistic = test_stat
                                    interpretation = "Normal distribution" if p_value > alpha else "Non-normal distribution"
                                else:
                                    test_stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                                    test_statistic = test_stat
                                    interpretation = "Normal distribution" if p_value > alpha else "Non-normal distribution"
                                    
                            elif test_type == "One-Sample T-Test":
                                test_value = settings.get('test_value', 0.0)
                                test_stat, p_value = stats.ttest_1samp(data, test_value)
                                test_statistic = test_stat
                                interpretation = f"Mean significantly different from {test_value}" if p_value < alpha else f"Mean not significantly different from {test_value}"
                            
                            if p_value is not None:
                                # Calculate Mean ± SD for the parameter in this system
                                mean_val = data.mean()
                                std_val = data.std()
                                mean_sd = f"{mean_val:.2f} ± {std_val:.2f}"
                                
                                system_data.append({
                                    'Parameter': column,
                                    'Mean ± SD': mean_sd,
                                    'Test_Statistic': round(test_statistic, 4) if test_statistic else None,
                                    'P_Value': round(p_value, 6),
                                    'Significant': 'Yes' if p_value < alpha else 'No',
                                    'Interpretation': interpretation
                                })
                                
                        except Exception as e:
                            continue
                    
                    if system_data:
                        results['system_pvalues'][master_col][system] = pd.DataFrame(system_data)
    
    return results
