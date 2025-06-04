"""
Web Data Utilities for Statistical Analysis Web Portal
This module provides functions for importing statistical reference data from the web
and comparing user data with standard reference values.
"""

import pandas as pd
import trafilatura
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from visualization_utils import apply_excel_theme

def get_website_text_content(url):
    """
    Extract the main text content from a website.
    
    Parameters:
    -----------
    url : str
        URL of the website to scrape
        
    Returns:
    --------
    str
        Extracted text content from the website
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text
    except Exception as e:
        return f"Error accessing website: {str(e)}"

def extract_statistical_data(text):
    """
    Extract statistical reference data from text content.
    
    Parameters:
    -----------
    text : str
        Text containing statistical references
        
    Returns:
    --------
    dict
        Dictionary of extracted statistical data
    """
    data = {}
    
    # Extract p-values, statistical thresholds, reference ranges
    try:
        # Look for p-value thresholds
        p_value_match = re.search(r'p\s*(?:<|=)\s*(0\.\d+)', text, re.IGNORECASE)
        if p_value_match:
            data['significance_threshold'] = float(p_value_match.group(1))
        
        # Look for reference ranges in format "Reference Range: X-Y"
        reference_ranges = re.findall(r'reference\s+range\s*:\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', 
                                     text, re.IGNORECASE)
        if reference_ranges:
            data['reference_ranges'] = [{'min': float(r[0]), 'max': float(r[1])} for r in reference_ranges]
        
        # Look for mean values in format "Mean: X" or "Average: X"
        mean_match = re.search(r'(mean|average)\s*:\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if mean_match:
            data['reference_mean'] = float(mean_match.group(2))
            
        # Look for standard deviation values in format "SD: X" or "Standard Deviation: X"
        sd_match = re.search(r'(sd|standard\s+deviation)\s*:\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if sd_match:
            data['reference_std'] = float(sd_match.group(2))
    
    except Exception as e:
        data['error'] = f"Error extracting statistical data: {str(e)}"
    
    return data

def parse_statistical_table(text):
    """
    Attempt to parse a statistical table from text.
    
    Parameters:
    -----------
    text : str
        Text that may contain a statistical table
        
    Returns:
    --------
    pd.DataFrame or None
        Extracted table as a DataFrame, or None if no table found
    """
    # Look for table-like structures in the text
    try:
        # Simple row-based table detection
        rows = []
        lines = text.split('\n')
        
        # Find potential table header
        header_idx = -1
        for i, line in enumerate(lines):
            # Look for lines with multiple separators like | or tabs, or multiple spaces
            if ('|' in line and line.count('|') > 2) or '\t' in line or re.search(r'\s{2,}', line):
                header_idx = i
                break
        
        if header_idx >= 0:
            # Try to determine the separator
            separator = None
            header = lines[header_idx]
            
            if '|' in header and header.count('|') > 2:
                separator = '|'
            elif '\t' in header:
                separator = '\t'
            else:
                # Try to split on multiple spaces
                separator = 'space'
            
            # Process header and rows
            if separator == 'space':
                # Split on multiple spaces
                header_parts = re.split(r'\s{2,}', header.strip())
                headers = [h.strip() for h in header_parts if h.strip()]
                
                # Process data rows
                for i in range(header_idx + 1, min(header_idx + 20, len(lines))):
                    if not lines[i].strip():
                        continue
                    
                    row_parts = re.split(r'\s{2,}', lines[i].strip())
                    row = [p.strip() for p in row_parts if p.strip()]
                    
                    if len(row) >= len(headers) * 0.7:  # If at least 70% of columns are present
                        # Pad row if necessary
                        while len(row) < len(headers):
                            row.append('')
                        rows.append(row[:len(headers)])  # Truncate if too long
            else:
                # Split on separator character
                header_parts = header.split(separator)
                headers = [h.strip() for h in header_parts if h.strip()]
                
                # Process data rows
                for i in range(header_idx + 1, min(header_idx + 20, len(lines))):
                    if not lines[i].strip() or separator not in lines[i]:
                        continue
                    
                    row_parts = lines[i].split(separator)
                    row = [p.strip() for p in row_parts if p.strip()]
                    
                    if len(row) >= len(headers) * 0.7:  # If at least 70% of columns are present
                        # Pad row if necessary
                        while len(row) < len(headers):
                            row.append('')
                        rows.append(row[:len(headers)])  # Truncate if too long
            
            if rows:
                # Create DataFrame - ensure headers and rows are compatible
                # This fixes any type issues with headers/columns mismatch
                headers_clean = [str(h) for h in headers]  # Convert headers to strings
                df = pd.DataFrame(rows, columns=headers_clean)
                
                # Convert numeric columns
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass  # Leave as string if conversion fails
                
                return df
    
    except Exception as e:
        st.error(f"Error parsing table: {str(e)}")
    
    return None

def compare_with_reference(user_data, reference_data, column_name):
    """
    Compare user data with reference values.
    
    Parameters:
    -----------
    user_data : pd.DataFrame
        User's dataset
    reference_data : dict
        Reference data with statistical values
    column_name : str
        Column in user_data to compare
        
    Returns:
    --------
    dict
        Comparison results
    """
    results = {
        'column': column_name,
        'in_range_count': 0,
        'in_range_percent': 0,
        'out_of_range_count': 0,
        'out_of_range_percent': 0,
        'too_low_count': 0,
        'too_high_count': 0
    }
    
    # Extract user data statistics
    user_stats = {
        'mean': user_data[column_name].mean(),
        'std': user_data[column_name].std(),
        'min': user_data[column_name].min(),
        'max': user_data[column_name].max(),
        'count': user_data[column_name].count()
    }
    
    results['user_stats'] = user_stats
    
    # Compare with reference ranges if available
    if 'reference_ranges' in reference_data and reference_data['reference_ranges']:
        ref_range = reference_data['reference_ranges'][0]  # Use first reference range
        min_val, max_val = ref_range['min'], ref_range['max']
        
        in_range = user_data[(user_data[column_name] >= min_val) & 
                           (user_data[column_name] <= max_val)]
        too_low = user_data[user_data[column_name] < min_val]
        too_high = user_data[user_data[column_name] > max_val]
        
        results['in_range_count'] = len(in_range)
        results['in_range_percent'] = len(in_range) / len(user_data) * 100 if len(user_data) > 0 else 0
        results['out_of_range_count'] = len(too_low) + len(too_high)
        results['out_of_range_percent'] = results['out_of_range_count'] / len(user_data) * 100 if len(user_data) > 0 else 0
        results['too_low_count'] = len(too_low)
        results['too_high_count'] = len(too_high)
    
    # Compare means if available
    if 'reference_mean' in reference_data and 'reference_std' in reference_data:
        ref_mean = reference_data['reference_mean']
        ref_std = reference_data['reference_std']
        
        results['reference_stats'] = {
            'mean': ref_mean,
            'std': ref_std
        }
        
        # Calculate z-score
        results['z_score'] = (user_stats['mean'] - ref_mean) / ref_std if ref_std != 0 else 0
    
    return results

def visualize_comparison(user_data, reference_data, column_name):
    """
    Visualize comparison between user data and reference values.
    
    Parameters:
    -----------
    user_data : pd.DataFrame
        User's dataset
    reference_data : dict
        Reference data with statistical values
    column_name : str
        Column in user_data to compare
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Visualization figure
    """
    # Create distribution plot with reference ranges
    fig = go.Figure()
    
    # Add user data histogram
    fig.add_trace(go.Histogram(
        x=user_data[column_name],
        name="User Data",
        opacity=0.7,
        nbinsx=30,
        marker=dict(color="rgba(0, 114, 178, 0.7)")
    ))
    
    # Add reference ranges if available
    if 'reference_ranges' in reference_data and reference_data['reference_ranges']:
        ref_range = reference_data['reference_ranges'][0]  # Use first reference range
        min_val, max_val = ref_range['min'], ref_range['max']
        
        # Add reference range as shaded region
        fig.add_vrect(
            x0=min_val,
            x1=max_val,
            fillcolor="rgba(0, 158, 115, 0.2)",
            layer="below",
            line_width=0,
            annotation_text="Reference Range",
            annotation_position="top left",
            annotation=dict(font=dict(family="Times New Roman, Times, serif"))
        )
    
    # Add reference mean if available
    if 'reference_mean' in reference_data:
        fig.add_vline(
            x=reference_data['reference_mean'],
            line_dash="dash",
            line_color="rgba(230, 159, 0, 0.8)",
            line_width=2,
            annotation=dict(
                text="Reference Mean",
                font=dict(family="Times New Roman, Times, serif")
            )
        )
    
    # Update layout with Excel-like styling
    fig.update_layout(
        title=f"Distribution of {column_name} with Reference Values",
        xaxis_title=column_name,
        yaxis_title="Frequency",
        template="plotly_white",
        font=dict(family="Times New Roman, Times, serif"),
        title_font=dict(family="Times New Roman, Times, serif", size=14),
        legend=dict(font=dict(family="Times New Roman, Times, serif"))
    )
    
    # Apply Excel theme
    fig = apply_excel_theme(fig)
    
    return fig

def reference_data_ui():
    """
    Display the reference data import and comparison UI
    """
    st.write("## Statistical Reference Data")
    st.write("Import statistical reference values from websites to compare with your data.")
    
    # URL input
    url = st.text_input("Enter URL containing statistical reference data:", 
                     placeholder="https://example.com/statistical-references")
    
    if url:
        with st.spinner("Retrieving data from website..."):
            # Get content from website
            content = get_website_text_content(url)
            
            if content and not content.startswith("Error"):
                # Display content preview
                st.write("### Content Preview")
                st.write(content[:500] + "..." if len(content) > 500 else content)
                
                # Extract statistical data
                ref_data = extract_statistical_data(content)
                
                # Parse tables if present
                ref_table = parse_statistical_table(content)
                
                # Display extracted reference data
                st.write("### Extracted Reference Data")
                
                if ref_data and any(k != 'error' for k in ref_data.keys()):
                    # Show extracted values
                    cols = st.columns(2)
                    
                    with cols[0]:
                        if 'significance_threshold' in ref_data:
                            st.metric("Significance Threshold (p-value)", 
                                    f"p < {ref_data['significance_threshold']}")
                            
                        if 'reference_mean' in ref_data:
                            st.metric("Reference Mean", f"{ref_data['reference_mean']:.3f}")
                    
                    with cols[1]:
                        if 'reference_ranges' in ref_data and ref_data['reference_ranges']:
                            range_val = ref_data['reference_ranges'][0]
                            st.metric("Reference Range", 
                                    f"{range_val['min']:.1f} - {range_val['max']:.1f}")
                            
                        if 'reference_std' in ref_data:
                            st.metric("Reference Std Dev", f"{ref_data['reference_std']:.3f}")
                
                if ref_table is not None:
                    st.write("### Extracted Reference Table")
                    st.dataframe(ref_table.style.format("{:.3f}", subset=ref_table.select_dtypes('number').columns))
                
                # Compare with user data if available
                if 'data' in st.session_state and st.session_state.data is not None:
                    user_df = st.session_state.data
                    
                    st.write("### Compare Your Data with References")
                    
                    # Select column for comparison
                    numeric_cols = user_df.select_dtypes(include=['number']).columns.tolist()
                    
                    if numeric_cols:
                        selected_col = st.selectbox("Select column to compare:", numeric_cols)
                        
                        if selected_col:
                            # Perform comparison
                            comparison = compare_with_reference(user_df, ref_data, selected_col)
                            
                            # Display comparison results
                            st.write("#### Comparison Results")
                            
                            # Key metrics
                            metrics = st.columns(3)
                            
                            with metrics[0]:
                                if 'reference_stats' in comparison:
                                    ref_mean = comparison['reference_stats']['mean']
                                    user_mean = comparison['user_stats']['mean']
                                    diff = user_mean - ref_mean
                                    st.metric("Mean Comparison", 
                                             f"{user_mean:.2f}", 
                                             f"{diff:+.2f} from reference")
                            
                            with metrics[1]:
                                if 'in_range_percent' in comparison and comparison['in_range_percent'] > 0:
                                    st.metric("Values in Range", 
                                             f"{comparison['in_range_percent']:.1f}%",
                                             f"{comparison['in_range_count']} values")
                            
                            with metrics[2]:
                                if 'z_score' in comparison:
                                    z = comparison['z_score']
                                    st.metric("Z-Score", 
                                             f"{z:.2f}",
                                             "Within normal range" if abs(z) < 1.96 else "Outside normal range")
                            
                            # Visualization
                            st.write("#### Visualization")
                            fig = visualize_comparison(user_df, ref_data, selected_col)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numeric columns found in your data for comparison.")
                else:
                    st.info("Upload your data first to compare with reference values.")
            else:
                st.error(content if content else "Unable to retrieve data from the website.")