"""
AI Data Preparation Utilities for Statistical Analysis Web Portal
This module prepares and consolidates data from the application for AI analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


def get_available_analysis_results() -> List[str]:
    """
    Scan session state to identify which analyses have been performed and have results stored
    
    Returns:
    --------
    List[str]
        List of names of available analyses that can be included in AI summary
    """
    available_analyses = []
    
    # Debug: Print all session state keys for troubleshooting
    print("Session state keys:", list(st.session_state.keys()))
    
    # Check for descriptive statistics - if data is loaded and processed, descriptive stats are available
    if 'data' in st.session_state and st.session_state.data is not None:
        available_analyses.append("Descriptive Statistics")
    
    # Check for t-test results
    if ('ttest_results' in st.session_state or 
        ('analysis_results' in st.session_state and 
         isinstance(st.session_state.analysis_results, dict) and
         st.session_state.analysis_results.get('type') == 'ttest')):
        available_analyses.append("t-Test Analysis")
    
    # Check for ANOVA results
    if ('anova_results' in st.session_state or 
        ('analysis_results' in st.session_state and 
         isinstance(st.session_state.analysis_results, dict) and
         st.session_state.analysis_results.get('type') == 'anova')):
        available_analyses.append("ANOVA Analysis")
    
    # Check for correlation analysis
    if ('correlation_results' in st.session_state or 
        ('analysis_results' in st.session_state and 
         isinstance(st.session_state.analysis_results, dict) and
         st.session_state.analysis_results.get('type') == 'correlation')):
        available_analyses.append("Correlation Analysis")
    
    # Check for p-value analysis - check if p-value analysis has been performed
    pvalue_performed = False
    if 'data' in st.session_state and st.session_state.data is not None:
        # Check if any numerical columns exist (p-value analysis can be performed)
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            pvalue_performed = True
    
    # Also check specific p-value result storage
    if ('pvalue_results' in st.session_state or 
        'pvalue_analysis_results' in st.session_state or
        'pvalue_data' in st.session_state or
        pvalue_performed):
        available_analyses.append("P-Value Analysis")
    
    # Check for chi-square test results
    if ('chi_square_results' in st.session_state or 
        ('analysis_results' in st.session_state and 
         isinstance(st.session_state.analysis_results, dict) and
         st.session_state.analysis_results.get('type') == 'chi_square')):
        available_analyses.append("Chi-Square Test")
    
    # Check for multi-sheet analysis
    if 'multi_sheet_analysis' in st.session_state:
        available_analyses.append("Multi-Sheet Analysis")
    
    # Check for machine learning results
    if 'ml_results' in st.session_state:
        available_analyses.append("Machine Learning Analysis")
    
    return available_analyses


def format_data_for_ai(selected_analyses_names: List[str]) -> Optional[str]:
    """
    Retrieve and format selected analysis data into a consolidated string for AI processing
    
    Parameters:
    -----------
    selected_analyses_names : List[str]
        List of analysis names selected by the user for AI summary
        
    Returns:
    --------
    Optional[str]
        Consolidated, well-formatted string containing all selected analysis results,
        or None if no valid data is found
    """
    if not selected_analyses_names:
        return None
    
    consolidated_text = ""
    data_found = False
    
    # Get basic dataset information
    if 'data' in st.session_state and st.session_state.data is not None:
        df = st.session_state.data
        consolidated_text += f"""
DATASET OVERVIEW:
- Dataset Name: {st.session_state.get('filename', 'Unknown')}
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Missing Values: {df.isnull().sum().sum()} total missing values
- Column Names: {', '.join(df.columns.tolist())}

"""
        data_found = True
    
    # Process each selected analysis
    for analysis_name in selected_analyses_names:
        
        if analysis_name == "Descriptive Statistics":
            desc_text = _format_descriptive_statistics()
            if desc_text:
                consolidated_text += desc_text
                data_found = True
        
        elif analysis_name == "t-Test Analysis":
            ttest_text = _format_ttest_results()
            if ttest_text:
                consolidated_text += ttest_text
                data_found = True
        
        elif analysis_name == "ANOVA Analysis":
            anova_text = _format_anova_results()
            if anova_text:
                consolidated_text += anova_text
                data_found = True
        
        elif analysis_name == "Correlation Analysis":
            corr_text = _format_correlation_results()
            if corr_text:
                consolidated_text += corr_text
                data_found = True
        
        elif analysis_name == "P-Value Analysis":
            pvalue_text = _format_pvalue_results()
            if pvalue_text:
                consolidated_text += pvalue_text
                data_found = True
        
        elif analysis_name == "Chi-Square Test":
            chi_text = _format_chi_square_results()
            if chi_text:
                consolidated_text += chi_text
                data_found = True
        
        elif analysis_name == "Multi-Sheet Analysis":
            multi_text = _format_multi_sheet_analysis()
            if multi_text:
                consolidated_text += multi_text
                data_found = True
        
        elif analysis_name == "Machine Learning Analysis":
            ml_text = _format_ml_results()
            if ml_text:
                consolidated_text += ml_text
                data_found = True
    
    return consolidated_text if data_found else None


def _format_descriptive_statistics() -> Optional[str]:
    """Format descriptive statistics results for AI analysis"""
    try:
        # Generate descriptive statistics from current data
        if 'data' in st.session_state and st.session_state.data is not None:
            df = st.session_state.data
            
            # Get descriptive statistics for numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
                
            stats_df = df[numeric_cols].describe()
            
            text = "\nDESCRIPTIVE STATISTICS:\n"
            text += "=" * 50 + "\n"
            text += stats_df.to_string() + "\n\n"
            
            # Add detailed interpretation
            text += "Key Statistical Insights:\n"
            for col in numeric_cols:
                text += f"\n{col}:\n"
                text += f"  - Mean: {stats_df.loc['mean', col]:.3f}\n"
                text += f"  - Standard Deviation: {stats_df.loc['std', col]:.3f}\n"
                text += f"  - Range: {stats_df.loc['min', col]:.3f} to {stats_df.loc['max', col]:.3f}\n"
                text += f"  - Median: {stats_df.loc['50%', col]:.3f}\n"
                text += f"  - IQR: {stats_df.loc['25%', col]:.3f} to {stats_df.loc['75%', col]:.3f}\n"
            
            # Add data quality information
            text += "\nData Quality Assessment:\n"
            for col in numeric_cols:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                text += f"  - {col}: {missing_count} missing values ({missing_pct:.1f}%)\n"
            
            return text
    
    except Exception as e:
        print(f"Error formatting descriptive statistics: {str(e)}")
    
    return None


def _format_ttest_results() -> Optional[str]:
    """Format t-test results for AI analysis"""
    return None


def _format_anova_results() -> Optional[str]:
    """Format ANOVA results for AI analysis"""
    return None


def _format_correlation_results() -> Optional[str]:
    """Format correlation analysis results for AI analysis"""
    return None


def _format_pvalue_results() -> Optional[str]:
    """Format p-value analysis results for AI analysis"""
    try:
        # Generate p-value analysis from current data
        if 'data' in st.session_state and st.session_state.data is not None:
            df = st.session_state.data
            
            # Get numerical columns for p-value analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return None
            
            text = "\nP-VALUE ANALYSIS:\n"
            text += "=" * 50 + "\n"
            
            # Perform basic statistical tests and extract p-values
            from scipy import stats
            
            # Normality tests
            text += "Normality Tests (Shapiro-Wilk):\n"
            for col in numeric_cols:
                if len(df[col].dropna()) >= 3:  # Minimum sample size for Shapiro-Wilk
                    stat, p_value = stats.shapiro(df[col].dropna())
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    text += f"  - {col}: p-value = {p_value:.6f} ({significance})\n"
            
            # Correlation p-values (if more than one numeric column)
            if len(numeric_cols) >= 2:
                text += "\nCorrelation Significance Tests:\n"
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        if len(df[[col1, col2]].dropna()) >= 3:
                            corr_coef, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                            significance = "Significant" if p_value < 0.05 else "Not Significant"
                            text += f"  - {col1} vs {col2}: r = {corr_coef:.3f}, p-value = {p_value:.6f} ({significance})\n"
            
            # Add interpretation
            text += "\nStatistical Significance Interpretation:\n"
            text += "  - p < 0.05: Statistically significant result\n"
            text += "  - p ≥ 0.05: Not statistically significant\n"
            text += "  - Lower p-values indicate stronger evidence against null hypothesis\n"
            
            return text
    
    except Exception as e:
        print(f"Error formatting p-value analysis: {str(e)}")
    
    return None


def _format_chi_square_results() -> Optional[str]:
    """Format chi-square test results for AI analysis"""
    return None


def _format_multi_sheet_analysis() -> Optional[str]:
    """Format multi-sheet analysis results for AI analysis"""
    return None


def _format_ml_results() -> Optional[str]:
    """Format machine learning analysis results for AI analysis"""
    return None