"""
Machine Learning Utilities for Statistical Analysis Web Portal
This module provides automated data analysis and machine learning capabilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def analyze_data_structure(df):
    """
    Analyze the data structure and recommend appropriate analyses
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
        
    Returns:
    --------
    dict
        Dictionary containing data characteristics and suggested analyses
    """
    result = {
        'num_rows': len(df),
        'num_cols': len(df.columns),
        'suggested_analyses': []
    }
    
    # Categorize columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    result['numeric_columns'] = num_cols
    result['categorical_columns'] = cat_cols
    
    # Check for time-series data (look for date/time columns)
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' or 
                ('date' in col.lower() and df[col].dtype == 'object')]
    result['date_columns'] = date_cols
    
    # Check for pre/post columns and automatically pair them if possible
    pre_cols = [col for col in df.columns if 'pre' in col.lower()]
    post_cols = [col for col in df.columns if 'post' in col.lower()]
    has_pre_post = len(pre_cols) > 0 and len(post_cols) > 0
    result['has_pre_post_data'] = has_pre_post
    
    # Find matching pre/post pairs
    pre_post_pairs = []
    if has_pre_post:
        for pre_col in pre_cols:
            pre_base = pre_col.lower().replace('pre', '').strip()
            for post_col in post_cols:
                post_base = post_col.lower().replace('post', '').strip()
                if pre_base == post_base or similar_enough(pre_base, post_base):
                    pre_post_pairs.append({
                        'pre': pre_col,
                        'post': post_col,
                        'base_name': pre_base
                    })
        result['pre_post_pairs'] = pre_post_pairs
    
    # Identify possible grouping columns (categorical with few unique values)
    possible_group_cols = []
    for col in cat_cols:
        if df[col].nunique() < 10:  # Categorical with fewer than 10 unique values
            possible_group_cols.append(col)
    # Also include low-cardinality numeric cols as potential grouping variables
    for col in num_cols:
        if df[col].nunique() < 10:
            possible_group_cols.append(col)
    
    result['possible_grouping_columns'] = possible_group_cols
    
    # Identify likely system/group identifier columns - often these are the most important grouping variables
    system_cols = []
    for col in cat_cols:
        col_lower = str(col).lower()
        if 'system' in col_lower or 'type' in col_lower or 'group' in col_lower or 'category' in col_lower:
            if df[col].nunique() <= 5:  # System types are usually few
                system_cols.append(col)
    result['system_columns'] = system_cols
    
    # Detect percentage columns and calculations
    pct_cols = [col for col in df.columns if '%' in str(col) or 'percent' in str(col).lower() or 'reduction' in str(col).lower()]
    result['percentage_columns'] = pct_cols
    
    # Identify input, process, and output columns
    input_cols, process_cols, output_cols = identify_column_roles(df)
    result['input_columns'] = input_cols
    result['process_columns'] = process_cols
    result['output_columns'] = output_cols
    
    # Detect calculation patterns
    calc_patterns = detect_calculation_patterns(df)
    result['calculation_patterns'] = calc_patterns
    
    # Suggest appropriate analyses based on data characteristics
    if len(num_cols) >= 2:
        result['suggested_analyses'].append('correlation_analysis')
    
    if len(possible_group_cols) > 0 and len(num_cols) > 0:
        result['suggested_analyses'].append('group_comparison')
        result['suggested_analyses'].append('anova')
    
    if has_pre_post:
        result['suggested_analyses'].append('paired_t_test')
        if len(pre_post_pairs) > 0:
            result['suggested_analyses'].append('percentage_reduction_analysis')
    
    if len(num_cols) >= 2:
        result['suggested_analyses'].append('regression_analysis')
    
    if len(pct_cols) > 0:
        result['suggested_analyses'].append('percentage_analysis')
    
    if len(num_cols) >= 2:
        result['suggested_analyses'].append('clustering')
    
    # If we have many numeric columns, suggest dimensionality reduction
    if len(num_cols) > 5:
        result['suggested_analyses'].append('dimensionality_reduction')
    
    # If we have system columns and pre/post data, suggest system comparison
    if len(system_cols) > 0 and has_pre_post:
        result['suggested_analyses'].append('system_comparison')
    
    return result

def similar_enough(str1, str2, threshold=0.7):
    """Check if two strings are similar enough using simple comparison"""
    # Convert to lowercase and remove common punctuation and spacing
    clean1 = str1.lower().replace(' ', '').replace('_', '').replace('-', '').replace('(', '').replace(')', '')
    clean2 = str2.lower().replace(' ', '').replace('_', '').replace('-', '').replace('(', '').replace(')', '')
    
    # Check for substring
    if clean1 in clean2 or clean2 in clean1:
        return True
    
    # Calculate similarity
    if len(clean1) == 0 or len(clean2) == 0:
        return False
    
    # Simple character matching
    common = sum(1 for c in clean1 if c in clean2)
    similarity = common / max(len(clean1), len(clean2))
    return similarity >= threshold

def identify_column_roles(df):
    """Identify columns as inputs, processing steps, or outputs based on patterns"""
    input_cols = []
    process_cols = []
    output_cols = []
    
    for col in df.columns:
        col_str = str(col).lower()
        
        # Input columns often have these characteristics
        if any(word in col_str for word in ['id', 'name', 'system', 'patient', 'age', 'gender', 'group']):
            input_cols.append(col)
            continue
            
        # Pre-treatment measurements are usually inputs
        if 'pre' in col_str and not any(word in col_str for word in ['diff', 'change', 'delta', 'reduction']):
            input_cols.append(col)
            continue
            
        # Output columns often have these characteristics
        if any(word in col_str for word in ['result', 'output', 'final', 'total']):
            output_cols.append(col)
            continue
            
        # Post-treatment measurements are usually outputs
        if 'post' in col_str:
            output_cols.append(col)
            continue
            
        # Percentages and reductions are usually outputs
        if any(word in col_str for word in ['%', 'percent', 'reduction', 'efficiency']):
            output_cols.append(col)
            continue
            
        # Processing columns have these characteristics
        if any(word in col_str for word in ['diff', 'delta', 'change', 'rate', 'time', 'duration', 'flow']):
            process_cols.append(col)
            continue
            
        # For remaining numeric columns, classify based on position and content
        if pd.api.types.is_numeric_dtype(df[col]):
            # If it has many unique values, it's likely a measurement (input or output)
            if df[col].nunique() > 10:
                # By default, put remaining numerics in process columns
                process_cols.append(col)
        else:
            # For remaining categorical columns, classify as inputs by default
            input_cols.append(col)
    
    # Remove duplicates while maintaining order
    input_cols = list(dict.fromkeys(input_cols))
    process_cols = list(dict.fromkeys(process_cols))
    output_cols = list(dict.fromkeys(output_cols))
    
    return input_cols, process_cols, output_cols

def detect_calculation_patterns(df):
    """Detect common calculation patterns in the data"""
    patterns = []
    
    # Get numeric columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Detect pre/post pairs and possible calculations
    pre_cols = [col for col in num_cols if 'pre' in str(col).lower()]
    post_cols = [col for col in num_cols if 'post' in str(col).lower()]
    
    # For each pre column, find matching post column and look for difference/percentage columns
    for pre_col in pre_cols:
        pre_base = str(pre_col).lower().replace('pre', '').strip()
        for post_col in post_cols:
            post_base = str(post_col).lower().replace('post', '').strip()
            if pre_base == post_base or similar_enough(pre_base, post_base):
                # Found a pre/post pair
                patterns.append({
                    'type': 'pre_post_comparison',
                    'pre_column': pre_col,
                    'post_column': post_col,
                    'base_name': pre_base
                })
                
                # Look for difference column
                diff_keywords = ['diff', 'delta', 'change']
                diff_cols = [col for col in num_cols if any(kw in str(col).lower() for kw in diff_keywords) and pre_base in str(col).lower()]
                
                if diff_cols:
                    for diff_col in diff_cols:
                        patterns.append({
                            'type': 'difference_calculation',
                            'input_columns': [pre_col, post_col],
                            'output_column': diff_col,
                            'base_name': pre_base
                        })
                
                # Look for percentage reduction column
                pct_keywords = ['%', 'percent', 'reduction', 'change']
                pct_cols = [col for col in num_cols if any(kw in str(col).lower() for kw in pct_keywords) and pre_base in str(col).lower()]
                
                if pct_cols:
                    for pct_col in pct_cols:
                        patterns.append({
                            'type': 'percentage_calculation',
                            'input_columns': [pre_col, post_col],
                            'output_column': pct_col,
                            'base_name': pre_base
                        })
    
    # Detect aggregation columns (sum, avg, etc.)
    agg_keywords = {
        'sum': ['sum', 'total'],
        'average': ['avg', 'average', 'mean'],
        'ratio': ['ratio', 'quotient', 'rate']
    }
    
    for agg_type, keywords in agg_keywords.items():
        for col in num_cols:
            col_str = str(col).lower()
            if any(kw in col_str for kw in keywords):
                patterns.append({
                    'type': f'{agg_type}_calculation',
                    'output_column': col,
                    'calculation': agg_type
                })
    
    return patterns

def prepare_data_for_ml(df, target_col=None, task_type='auto'):
    """
    Prepare data for machine learning by handling missing values,
    encoding categorical variables, and scaling numeric features
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to prepare
    target_col : str or None
        Target column name for supervised learning
    task_type : str
        'auto', 'regression', 'classification', or 'clustering'
        
    Returns:
    --------
    dict
        Dictionary with prepared data and preprocessing pipeline
    """
    # Make a copy to avoid modifying the original df
    data = df.copy()
    
    # Identify numeric and categorical columns
    num_cols = data.select_dtypes(include=['number']).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Separate target if provided
    X = data
    y = None
    
    if target_col is not None and target_col in data.columns:
        if target_col in num_cols:
            num_cols.remove(target_col)
        elif target_col in cat_cols:
            cat_cols.remove(target_col)
        
        y = data[target_col]
        X = data.drop(columns=[target_col])
    
    # Automatically determine task type if set to 'auto'
    if task_type == 'auto' and target_col is not None:
        if y.dtype in ['int64', 'float64'] and y.nunique() > 10:
            task_type = 'regression'
        else:
            task_type = 'classification'
    elif task_type == 'auto':
        task_type = 'clustering'  # Default to clustering if no target
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    # Return prepared data and preprocessing info
    return {
        'X': X,
        'y': y,
        'preprocessor': preprocessor,
        'task_type': task_type,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }

def train_model(prepared_data, model_type=None, params=None):
    """
    Train a machine learning model based on the prepared data
    
    Parameters:
    -----------
    prepared_data : dict
        Output from prepare_data_for_ml function
    model_type : str or None
        Type of model to train (if None, will be selected based on task_type)
    params : dict or None
        Model hyperparameters
        
    Returns:
    --------
    dict
        Dictionary with trained model and evaluation metrics
    """
    X = prepared_data['X']
    y = prepared_data['y']
    preprocessor = prepared_data['preprocessor']
    task_type = prepared_data['task_type']
    
    # Automatically select model if not specified
    if model_type is None:
        if task_type == 'regression':
            model_type = 'linear_regression'
        elif task_type == 'classification':
            model_type = 'random_forest'
        else:  # clustering
            model_type = 'kmeans'
    
    # Initialize model based on type
    model = None
    if model_type == 'linear_regression':
        model = LinearRegression(**(params or {}))
    elif model_type == 'random_forest_regression':
        model = RandomForestRegressor(n_estimators=100, **(params or {}))
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, **(params or {}))
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, **(params or {}))
    elif model_type == 'kmeans':
        n_clusters = params.get('n_clusters', 3) if params else 3
        model = KMeans(n_clusters=n_clusters)
    
    # Create and fit pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Handle different ML tasks
    result = {'model_pipeline': pipeline, 'task_type': task_type}
    
    if task_type in ['regression', 'classification'] and y is not None:
        # Split data for supervised learning
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate model
        if task_type == 'regression':
            result['metrics'] = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            # Add cross-validation scores
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
            result['cv_scores'] = {
                'mean_r2': cv_scores.mean(),
                'std_r2': cv_scores.std()
            }
            
            # Feature importance for RandomForest
            if model_type == 'random_forest_regression':
                feature_names = (
                    prepared_data['num_cols'] + 
                    list(pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names_out(prepared_data['cat_cols']))
                )
                importances = pipeline.named_steps['model'].feature_importances_
                result['feature_importance'] = dict(zip(feature_names, importances))
        
        elif task_type == 'classification':
            result['metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            # Add cross-validation scores
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
            result['cv_scores'] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std()
            }
            
            # Feature importance for RandomForest
            if model_type == 'random_forest':
                feature_names = (
                    prepared_data['num_cols'] + 
                    list(pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names_out(prepared_data['cat_cols']))
                )
                importances = pipeline.named_steps['model'].feature_importances_
                result['feature_importance'] = dict(zip(feature_names, importances))
    
    elif task_type == 'clustering':
        # For clustering, fit on all data
        pipeline.fit(X)
        
        # Get cluster labels
        cluster_labels = pipeline.named_steps['model'].labels_
        
        # Add cluster labels to original data for analysis
        result['cluster_labels'] = cluster_labels
        
        # Compute cluster metrics
        if len(prepared_data['num_cols']) > 0:
            cluster_centers = pipeline.named_steps['model'].cluster_centers_
            inertia = pipeline.named_steps['model'].inertia_
            
            result['metrics'] = {
                'inertia': inertia,
                'n_clusters': len(np.unique(cluster_labels))
            }
    
    return result

def perform_automated_analysis(df, analysis_type, options=None):
    """
    Perform automated statistical analysis based on analysis type
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    analysis_type : str
        Type of analysis to perform
    options : dict or None
        Additional options for the analysis
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    options = options or {}
    result = {'analysis_type': analysis_type}
    
    if analysis_type == 'correlation_analysis':
        # Perform correlation analysis on numeric columns
        num_df = df.select_dtypes(include=['number'])
        corr_method = options.get('method', 'pearson')
        
        if len(num_df.columns) > 1:
            corr_matrix = num_df.corr(method=corr_method)
            # Calculate p-values
            p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                                    index=corr_matrix.index, 
                                    columns=corr_matrix.columns)
            
            for i, col_i in enumerate(corr_matrix.columns):
                for j, col_j in enumerate(corr_matrix.columns):
                    if i != j:  # Skip diagonal
                        if corr_method == 'pearson':
                            r, p = stats.pearsonr(num_df[col_i].dropna(), num_df[col_j].dropna())
                        elif corr_method == 'spearman':
                            r, p = stats.spearmanr(num_df[col_i].dropna(), num_df[col_j].dropna())
                        else:  # kendall
                            r, p = stats.kendalltau(num_df[col_i].dropna(), num_df[col_j].dropna())
                        p_values.loc[col_i, col_j] = p
            
            result['correlation_matrix'] = corr_matrix
            result['p_values'] = p_values
            
            # Find significant correlations
            alpha = options.get('alpha', 0.05)
            significant_corrs = []
            
            for i, col_i in enumerate(corr_matrix.columns):
                for j, col_j in enumerate(corr_matrix.columns):
                    if i < j:  # Only look at upper triangle to avoid duplicates
                        r = corr_matrix.loc[col_i, col_j]
                        p = p_values.loc[col_i, col_j]
                        
                        if abs(r) > 0.3 and p < alpha:  # Moderate correlation and significant
                            significant_corrs.append({
                                'variable_1': col_i,
                                'variable_2': col_j,
                                'correlation': r,
                                'p_value': p,
                                'strength': 'strong' if abs(r) > 0.7 else 'moderate'
                            })
            
            result['significant_correlations'] = significant_corrs
    
    elif analysis_type == 'group_comparison':
        group_col = options.get('group_column')
        
        if group_col and group_col in df.columns:
            # Get numeric columns
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Exclude the group column if it's numeric
            if group_col in num_cols:
                num_cols.remove(group_col)
            
            # Compare means across groups
            group_stats = {}
            for col in num_cols:
                group_means = df.groupby(group_col)[col].mean()
                group_stds = df.groupby(group_col)[col].std()
                group_counts = df.groupby(group_col)[col].count()
                
                # Perform ANOVA
                groups = [df[df[group_col] == g][col].dropna() for g in df[group_col].unique()]
                groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_result = {'f_statistic': f_stat, 'p_value': p_value}
                except:
                    anova_result = {'f_statistic': np.nan, 'p_value': np.nan}
                
                group_stats[col] = {
                    'means': group_means.to_dict(),
                    'stds': group_stds.to_dict(),
                    'counts': group_counts.to_dict(),
                    'anova': anova_result
                }
            
            result['group_stats'] = group_stats
            result['group_column'] = group_col
    
    elif analysis_type == 'paired_t_test':
        pre_col = options.get('pre_column')
        post_col = options.get('post_column')
        
        if pre_col and post_col and pre_col in df.columns and post_col in df.columns:
            # Drop rows with missing values in either column
            valid_data = df.dropna(subset=[pre_col, post_col])
            
            if len(valid_data) > 1:  # Need at least 2 paired samples
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(valid_data[pre_col], valid_data[post_col])
                
                # Calculate mean difference and percent change
                mean_diff = valid_data[post_col].mean() - valid_data[pre_col].mean()
                if valid_data[pre_col].mean() != 0:
                    pct_change = (mean_diff / valid_data[pre_col].mean()) * 100
                else:
                    pct_change = np.nan
                
                result['t_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'mean_difference': mean_diff,
                    'percent_change': pct_change,
                    'pre_mean': valid_data[pre_col].mean(),
                    'post_mean': valid_data[post_col].mean(),
                    'pre_std': valid_data[pre_col].std(),
                    'post_std': valid_data[post_col].std(),
                    'n': len(valid_data)
                }
                
                # Check if there's a grouping column to do pre-post by group
                group_col = options.get('group_column')
                if group_col and group_col in df.columns:
                    group_results = {}
                    
                    for group_val in df[group_col].unique():
                        group_data = valid_data[valid_data[group_col] == group_val]
                        
                        if len(group_data) > 1:  # Need at least 2 paired samples per group
                            # Perform paired t-test for this group
                            t_stat, p_value = stats.ttest_rel(group_data[pre_col], group_data[post_col])
                            
                            # Calculate mean difference and percent change
                            mean_diff = group_data[post_col].mean() - group_data[pre_col].mean()
                            if group_data[pre_col].mean() != 0:
                                pct_change = (mean_diff / group_data[pre_col].mean()) * 100
                            else:
                                pct_change = np.nan
                            
                            group_results[group_val] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'mean_difference': mean_diff,
                                'percent_change': pct_change,
                                'pre_mean': group_data[pre_col].mean(),
                                'post_mean': group_data[post_col].mean(),
                                'pre_std': group_data[pre_col].std(),
                                'post_std': group_data[post_col].std(),
                                'n': len(group_data)
                            }
                    
                    result['group_t_tests'] = group_results
                    result['group_column'] = group_col
    
    elif analysis_type == 'regression_analysis':
        target_col = options.get('target_column')
        predictor_cols = options.get('predictor_columns')
        
        if target_col and target_col in df.columns:
            # If predictor columns not specified, use all numeric columns except target
            if not predictor_cols:
                predictor_cols = df.select_dtypes(include=['number']).columns.tolist()
                if target_col in predictor_cols:
                    predictor_cols.remove(target_col)
            
            # Ensure all specified predictors exist in dataframe
            predictor_cols = [col for col in predictor_cols if col in df.columns]
            
            if predictor_cols:
                # Prepare data for regression
                X = df[predictor_cols].copy()
                y = df[target_col].copy()
                
                # Drop rows with missing values
                valid_indices = ~(X.isna().any(axis=1) | y.isna())
                X = X[valid_indices]
                y = y[valid_indices]
                
                if len(X) > len(predictor_cols) + 1:  # Enough samples for regression
                    # Add constant for intercept
                    X_sm = sm.add_constant(X)
                    
                    # Fit OLS model using statsmodels
                    try:
                        model = sm.OLS(y, X_sm).fit()
                        
                        # Extract results
                        result['regression'] = {
                            'summary': {
                                'r_squared': model.rsquared,
                                'adj_r_squared': model.rsquared_adj,
                                'f_statistic': model.fvalue,
                                'f_pvalue': model.f_pvalue,
                                'n_observations': model.nobs,
                                'aic': model.aic,
                                'bic': model.bic
                            },
                            'coefficients': {},
                            'formula': f"{target_col} ~ {' + '.join(predictor_cols)}"
                        }
                        
                        # Get coefficient details
                        for i, predictor in enumerate(model.params.index):
                            var_name = predictor
                            if predictor == 'const':
                                var_name = 'Intercept'
                            
                            result['regression']['coefficients'][var_name] = {
                                'estimate': model.params[i],
                                'std_error': model.bse[i],
                                't_value': model.tvalues[i],
                                'p_value': model.pvalues[i],
                                'confidence_interval_95': [
                                    model.conf_int().iloc[i, 0],
                                    model.conf_int().iloc[i, 1]
                                ]
                            }
                        
                        # Linear Regression assumptions tests
                        # 1. Normality of residuals (Jarque-Bera test)
                        jb_test = sm.stats.jarque_bera(model.resid)
                        # 2. Heteroscedasticity (Breusch-Pagan test)
                        bp_test = sm.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
                        
                        result['regression']['diagnostics'] = {
                            'jarque_bera': {
                                'statistic': jb_test[0],
                                'p_value': jb_test[1]
                            },
                            'breusch_pagan': {
                                'statistic': bp_test[0],
                                'p_value': bp_test[1]
                            }
                        }
                    except Exception as e:
                        result['regression'] = {'error': str(e)}
    
    elif analysis_type == 'anova':
        dependent_var = options.get('dependent_variable')
        factor_vars = options.get('factor_variables')
        
        if dependent_var and dependent_var in df.columns:
            # If factor variables not specified, use all categorical columns
            if not factor_vars:
                factor_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
                # Also include low-cardinality numeric cols as potential factors
                for col in df.select_dtypes(include=['number']).columns:
                    if col != dependent_var and df[col].nunique() < 10:
                        factor_vars.append(col)
            
            # Ensure all specified factors exist in dataframe
            factor_vars = [col for col in factor_vars if col in df.columns]
            
            if factor_vars:
                result['anova_results'] = {}
                
                # Perform one-way ANOVA for each factor
                for factor in factor_vars:
                    # Drop rows with missing values
                    valid_data = df.dropna(subset=[dependent_var, factor])
                    
                    if len(valid_data) > 1:
                        groups = [valid_data[valid_data[factor] == val][dependent_var].values 
                                 for val in valid_data[factor].unique()]
                        groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                        
                        try:
                            # One-way ANOVA
                            f_stat, p_value = stats.f_oneway(*groups)
                            
                            # Group statistics
                            group_stats = valid_data.groupby(factor)[dependent_var].agg(['mean', 'std', 'count'])
                            
                            result['anova_results'][factor] = {
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'group_stats': group_stats.to_dict('index')
                            }
                            
                            # Try more detailed ANOVA with statsmodels
                            try:
                                # Create model formula
                                formula = f"{dependent_var} ~ C({factor})"
                                model = ols(formula, data=valid_data).fit()
                                anova_table = sm.stats.anova_lm(model, typ=2)
                                
                                result['anova_results'][factor]['detailed'] = {
                                    'df_factor': anova_table.iloc[0, 0],
                                    'df_residual': anova_table.iloc[1, 0],
                                    'sum_sq_factor': anova_table.iloc[0, 1],
                                    'sum_sq_residual': anova_table.iloc[1, 1],
                                    'mean_sq_factor': anova_table.iloc[0, 2],
                                    'mean_sq_residual': anova_table.iloc[1, 2],
                                    'f_value': anova_table.iloc[0, 3],
                                    'p_value': anova_table.iloc[0, 4]
                                }
                            except:
                                pass  # If detailed ANOVA fails, we still have basic results
                        except:
                            result['anova_results'][factor] = {
                                'error': 'Could not perform ANOVA'
                            }
                
                # Try two-way ANOVA if we have multiple factors
                if len(factor_vars) >= 2:
                    # Consider first two factors for simplicity
                    factor1, factor2 = factor_vars[0], factor_vars[1]
                    valid_data = df.dropna(subset=[dependent_var, factor1, factor2])
                    
                    try:
                        # Create model formula with interaction term
                        formula = f"{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
                        model = ols(formula, data=valid_data).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        
                        result['two_way_anova'] = {
                            'factors': [factor1, factor2],
                            'results': {
                                f'C({factor1})': {
                                    'df': anova_table.iloc[0, 0],
                                    'sum_sq': anova_table.iloc[0, 1],
                                    'mean_sq': anova_table.iloc[0, 2],
                                    'f_value': anova_table.iloc[0, 3],
                                    'p_value': anova_table.iloc[0, 4]
                                },
                                f'C({factor2})': {
                                    'df': anova_table.iloc[1, 0],
                                    'sum_sq': anova_table.iloc[1, 1],
                                    'mean_sq': anova_table.iloc[1, 2],
                                    'f_value': anova_table.iloc[1, 3],
                                    'p_value': anova_table.iloc[1, 4]
                                },
                                f'C({factor1}):C({factor2})': {
                                    'df': anova_table.iloc[2, 0],
                                    'sum_sq': anova_table.iloc[2, 1],
                                    'mean_sq': anova_table.iloc[2, 2],
                                    'f_value': anova_table.iloc[2, 3],
                                    'p_value': anova_table.iloc[2, 4]
                                }
                            }
                        }
                    except:
                        result['two_way_anova'] = {
                            'error': 'Could not perform two-way ANOVA'
                        }
    
    elif analysis_type == 'clustering':
        # Get numeric columns for clustering
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) >= 2:  # Need at least 2 dimensions for meaningful clustering
            # Prepare data, dropping rows with any missing values
            cluster_data = df[num_cols].dropna()
            
            if len(cluster_data) > 10:  # Need reasonable number of samples
                # Standardize data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Determine optimal number of clusters using elbow method
                max_clusters = min(10, len(cluster_data) // 5)  # Limit based on sample size
                inertias = []
                
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(scaled_data)
                    inertias.append(kmeans.inertia_)
                
                # Simple heuristic for elbow point detection
                optimal_k = 2  # Default to 2 clusters
                if len(inertias) > 2:
                    # Calculate rate of inertia decrease
                    decreases = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
                    
                    # Find where the rate of decrease slows down significantly
                    for i in range(1, len(decreases)):
                        if decreases[i] / decreases[0] < 0.2:  # Less than 20% of initial decrease
                            optimal_k = i + 2  # +2 because we started at k=2
                            break
                
                # Perform clustering with optimal k
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                # Add cluster centers
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                
                # Prepare cluster statistics
                cluster_stats = {}
                cluster_data['Cluster'] = cluster_labels
                
                for i in range(optimal_k):
                    cluster_members = cluster_data[cluster_data['Cluster'] == i]
                    
                    if len(cluster_members) > 0:
                        stats_dict = {}
                        for col in num_cols:
                            stats_dict[col] = {
                                'mean': cluster_members[col].mean(),
                                'std': cluster_members[col].std(),
                                'min': cluster_members[col].min(),
                                'max': cluster_members[col].max(),
                                'count': len(cluster_members)
                            }
                        
                        cluster_stats[f'Cluster {i+1}'] = stats_dict
                
                result['clustering'] = {
                    'method': 'kmeans',
                    'n_clusters': optimal_k,
                    'inertias': dict(zip(range(2, max_clusters + 1), inertias)),
                    'optimal_k': optimal_k,
                    'cluster_stats': cluster_stats,
                    'cluster_centers': dict(zip([f'Cluster {i+1}' for i in range(optimal_k)], 
                                              [dict(zip(num_cols, center)) for center in centers]))
                }
    
    elif analysis_type == 'descriptive_statistics':
        # Get numeric columns
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if num_cols:
            # Calculate descriptive statistics
            desc_stats = df[num_cols].describe().transpose()
            
            # Add additional statistics
            skewness = df[num_cols].skew()
            kurtosis = df[num_cols].kurtosis()
            missing = df[num_cols].isna().sum()
            missing_pct = (missing / len(df)) * 100
            
            desc_stats['skewness'] = skewness
            desc_stats['kurtosis'] = kurtosis
            desc_stats['missing'] = missing
            desc_stats['missing_pct'] = missing_pct
            
            result['descriptive_statistics'] = desc_stats.to_dict('index')
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            cat_stats = {}
            
            for col in cat_cols:
                value_counts = df[col].value_counts()
                top_n = min(10, len(value_counts))  # Limit to top 10 categories
                
                cat_stats[col] = {
                    'unique_values': df[col].nunique(),
                    'missing': df[col].isna().sum(),
                    'missing_pct': (df[col].isna().sum() / len(df)) * 100,
                    'frequency': value_counts.head(top_n).to_dict(),
                    'frequency_pct': (value_counts.head(top_n) / value_counts.sum() * 100).to_dict()
                }
            
            result['categorical_statistics'] = cat_stats
    
    return result