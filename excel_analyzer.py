import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
import base64
from openpyxl import load_workbook

def load_excel_file(uploaded_file):
    """
    Load an Excel file and extract all sheets
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The Excel file uploaded by the user
        
    Returns:
    --------
    dict, dict
        Dictionary of sheet names to dataframes,
        Dictionary of sheet names to metadata
    """
    # Load all sheets from the Excel file
    excel_data = {}
    sheet_metadata = {}
    
    # First load with pandas to get the dataframes
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names
    
    for sheet_name in sheet_names:
        excel_data[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Now use openpyxl to get metadata about charts and tables
    file_content = uploaded_file.getvalue()
    wb = load_workbook(io.BytesIO(file_content))
    
    for sheet_name in sheet_names:
        sheet = wb[sheet_name]
        # Get charts in this sheet
        charts = []
        for chart in sheet._charts:
            chart_type = type(chart).__name__
            charts.append(chart_type)
        
        # Get tables in this sheet
        tables = []
        if hasattr(sheet, 'tables'):
            for table_name in sheet.tables:
                table = sheet.tables[table_name]
                table_range = table.ref
                tables.append({
                    'name': table_name,
                    'range': table_range
                })
        
        # Get named ranges that might be used for calculations or charts
        named_ranges = []
        for named_range in wb.defined_names.definedName:
            if named_range.value.startswith(f"'{sheet_name}'!"):
                named_ranges.append({
                    'name': named_range.name,
                    'reference': named_range.value
                })
        
        # Store metadata for this sheet
        sheet_metadata[sheet_name] = {
            'charts': charts,
            'tables': tables,
            'named_ranges': named_ranges,
            'has_charts': len(charts) > 0,
            'has_tables': len(tables) > 0
        }
    
    return excel_data, sheet_metadata

def analyze_excel_sheet(df, sheet_name):
    """
    Analyze a single Excel sheet and generate statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing sheet data
    sheet_name : str
        Name of the sheet
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    results = {}
    
    # Basic info
    results['rows'] = len(df)
    results['columns'] = len(df.columns)
    
    # Column types
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
    
    results['numerical_columns'] = num_cols
    results['categorical_columns'] = cat_cols
    results['date_columns'] = date_cols
    
    # Missing values
    missing_values = df.isnull().sum()
    results['missing_values'] = {
        'total': missing_values.sum(),
        'percentage': (missing_values.sum() / (len(df) * len(df.columns)) * 100),
        'by_column': missing_values.to_dict()
    }
    
    # Basic statistics for numerical columns
    if num_cols:
        results['statistics'] = df[num_cols].describe().to_dict()
    
    # Check for likely aggregations (sums, averages) typically found in Excel sheets
    # Excel sheets often have sum rows or average calculations at the bottom
    try:
        # Check if last rows might be aggregations (common in Excel reports)
        last_row = df.iloc[-1:].select_dtypes(include=np.number)
        prev_row = df.iloc[-2:-1].select_dtypes(include=np.number)
        
        if not last_row.empty and not prev_row.empty:
            # Check if last row is significantly different from previous row
            # (might indicate a total/average row)
            diff = (last_row.values > prev_row.values * 1.5).any() or \
                   (last_row.values < prev_row.values * 0.5).any()
            
            if diff:
                results['possible_aggregation_row'] = len(df) - 1
    except:
        pass
    
    return results

def recreate_excel_charts(df, sheet_name, metadata):
    """
    Attempt to recreate charts from Excel based on data patterns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing sheet data
    sheet_name : str
        Name of the sheet
    metadata : dict
        Metadata for the sheet including chart info
        
    Returns:
    --------
    list
        List of plotly figures
    """
    charts = []
    
    if not metadata.get('has_charts', False):
        # Try to identify possible charts from data structure
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
        
        # Check for time series data (common in Excel)
        if date_cols and num_cols:
            date_col = date_cols[0]
            for num_col in num_cols[:3]:  # Limit to first 3 numerical columns
                fig = px.line(df, x=date_col, y=num_col, 
                              title=f"{num_col} over {date_col}")
                charts.append(('line', fig))
        
        # Check for categorical vs numerical data (common for bar charts in Excel)
        if cat_cols and num_cols:
            cat_col = cat_cols[0]
            num_col = num_cols[0]
            
            # Only proceed if there aren't too many categories
            if df[cat_col].nunique() <= 15:
                fig = px.bar(df, x=cat_col, y=num_col, 
                             title=f"{num_col} by {cat_col}")
                charts.append(('bar', fig))
        
        # Check for scatter plot potential (relationships between numerical variables)
        if len(num_cols) >= 2:
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1], 
                            title=f"{num_cols[1]} vs {num_cols[0]}")
            charts.append(('scatter', fig))
            
        # If there are 3+ numerical columns, try a correlation heatmap
        if len(num_cols) >= 3:
            corr_df = df[num_cols].corr()
            fig = px.imshow(corr_df, text_auto=True,
                           title="Correlation Heatmap")
            charts.append(('heatmap', fig))
    else:
        # Charts are known to exist, try to recreate based on chart types in metadata
        chart_types = metadata.get('charts', [])
        
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
        
        for chart_type in chart_types:
            # Map Excel chart types to Plotly chart types
            if 'Bar' in chart_type and cat_cols and num_cols:
                cat_col = cat_cols[0]
                for num_col in num_cols[:2]:
                    fig = px.bar(df, x=cat_col, y=num_col, 
                                title=f"{num_col} by {cat_col}")
                    charts.append(('bar', fig))
            
            elif 'Line' in chart_type:
                # Line charts usually have a category or date as x-axis
                x_col = None
                if date_cols:
                    x_col = date_cols[0]
                elif cat_cols:
                    x_col = cat_cols[0]
                
                if x_col and num_cols:
                    for num_col in num_cols[:2]:
                        fig = px.line(df, x=x_col, y=num_col, 
                                    title=f"{num_col} over {x_col}")
                        charts.append(('line', fig))
            
            elif 'Pie' in chart_type and cat_cols and num_cols:
                cat_col = cat_cols[0]
                num_col = num_cols[0]
                
                # Only proceed if there aren't too many categories
                if df[cat_col].nunique() <= 10:
                    fig = px.pie(df, names=cat_col, values=num_col, 
                                title=f"{num_col} by {cat_col}")
                    charts.append(('pie', fig))
            
            elif 'Scatter' in chart_type and len(num_cols) >= 2:
                fig = px.scatter(df, x=num_cols[0], y=num_cols[1], 
                                title=f"{num_cols[1]} vs {num_cols[0]}")
                charts.append(('scatter', fig))
    
    return charts

def recreate_excel_tables(df, sheet_name, metadata):
    """
    Recreate tables from Excel data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing sheet data
    sheet_name : str
        Name of the sheet
    metadata : dict
        Metadata for the sheet including table info
        
    Returns:
    --------
    list
        List of styled tables as HTML
    """
    tables = []
    
    # If tables are identified in the metadata
    if metadata.get('has_tables', False):
        table_info = metadata.get('tables', [])
        
        for table in table_info:
            # Try to extract the table using the range information
            # This is a simplification - in a real app we'd calculate the exact cell ranges
            tables.append(df.head(10).style.format(precision=2))
    else:
        # No tables explicitly defined, just use the dataframe itself
        tables.append(df.head(10).style.format(precision=2))
    
    return tables

def export_analysis_to_excel(workbook_data, analysis_results, charts):
    """
    Export analysis results to an Excel file
    
    Parameters:
    -----------
    workbook_data : dict
        Dictionary of sheet names to dataframes
    analysis_results : dict
        Dictionary of sheet names to analysis results
    charts : dict
        Dictionary of sheet names to chart figures
        
    Returns:
    --------
    bytes
        Excel file as bytes
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # First write original data
        for sheet_name, df in workbook_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
        # Create an analysis summary sheet
        summary_data = []
        for sheet_name, results in analysis_results.items():
            summary_data.append({
                'Sheet Name': sheet_name,
                'Rows': results.get('rows', 0),
                'Columns': results.get('columns', 0),
                'Numerical Columns': len(results.get('numerical_columns', [])),
                'Categorical Columns': len(results.get('categorical_columns', [])),
                'Missing Values': results.get('missing_values', {}).get('total', 0),
                'Missing Values %': f"{results.get('missing_values', {}).get('percentage', 0):.2f}%",
                'Has Charts': len(charts.get(sheet_name, [])) > 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Analysis Summary', index=False)
        
        # Create a sheet for analysis details for each original sheet
        for sheet_name, results in analysis_results.items():
            # Create a new sheet for analysis
            analysis_sheet_name = f"{sheet_name[:25]}_Analysis" if len(sheet_name) > 25 else f"{sheet_name}_Analysis"
            
            # Create a dataframe with the analysis results
            numerical_cols = results.get('numerical_columns', [])
            stats_data = []
            
            if numerical_cols:
                for col in numerical_cols:
                    if col in results.get('statistics', {}).get('mean', {}):
                        stats_data.append({
                            'Column': col,
                            'Mean': results['statistics']['mean'].get(col, np.nan),
                            'Std Dev': results['statistics']['std'].get(col, np.nan),
                            'Min': results['statistics']['min'].get(col, np.nan),
                            '25%': results['statistics']['25%'].get(col, np.nan),
                            'Median': results['statistics']['50%'].get(col, np.nan),
                            '75%': results['statistics']['75%'].get(col, np.nan),
                            'Max': results['statistics']['max'].get(col, np.nan),
                            'Missing': results['missing_values']['by_column'].get(col, 0)
                        })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name=analysis_sheet_name, index=False)
            
        # Chart data can't be embedded directly in Excel with xlsxwriter
        # We would need to use additional chart generation libraries
    
    return output.getvalue()

def display_excel_analyzer():
    """
    Main function to display the Excel analyzer interface
    """
    st.title("Excel Workbook Analyzer 📊")
    st.write("""
    This tool analyzes your Excel workbook and recreates the analysis, charts, tables, and graphs.
    Upload an Excel file with multiple sheets to get started.
    """)
    
    # File Upload
    uploaded_file = st.file_uploader("Upload Excel Workbook", type=['xlsx', 'xls'])
    
    if uploaded_file:
        try:
            # Load Excel file
            with st.spinner("Loading Excel workbook..."):
                workbook_data, sheet_metadata = load_excel_file(uploaded_file)
                
                if not workbook_data:
                    st.error("Could not load any sheets from the Excel file. Please check the format.")
                    return
            
            # Store in session state
            if 'workbook_data' not in st.session_state:
                st.session_state.workbook_data = workbook_data
                st.session_state.sheet_metadata = sheet_metadata
                
                # Analyze each sheet
                st.session_state.analysis_results = {}
                st.session_state.charts = {}
                st.session_state.tables = {}
                
                for sheet_name, df in workbook_data.items():
                    with st.spinner(f"Analyzing sheet: {sheet_name}..."):
                        st.session_state.analysis_results[sheet_name] = analyze_excel_sheet(df, sheet_name)
                        st.session_state.charts[sheet_name] = recreate_excel_charts(df, sheet_name, sheet_metadata.get(sheet_name, {}))
                        st.session_state.tables[sheet_name] = recreate_excel_tables(df, sheet_name, sheet_metadata.get(sheet_name, {}))
            
            # Display sheet selector
            sheet_names = list(workbook_data.keys())
            selected_sheet = st.selectbox("Select a sheet to analyze:", sheet_names)
            
            if selected_sheet:
                df = workbook_data[selected_sheet]
                sheet_meta = sheet_metadata.get(selected_sheet, {})
                analysis = st.session_state.analysis_results.get(selected_sheet, {})
                charts = st.session_state.charts.get(selected_sheet, [])
                tables = st.session_state.tables.get(selected_sheet, [])
                
                # Display tabs for different analysis views
                tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Analysis", "Charts", "Tables"])
                
                with tab1:
                    st.subheader(f"Data Preview: {selected_sheet}")
                    st.dataframe(df.head(100))
                    
                    st.download_button(
                        "Download Sheet as CSV",
                        df.to_csv(index=False).encode('utf-8'),
                        f"{selected_sheet}.csv",
                        "text/csv",
                        key=f"download_{selected_sheet}"
                    )
                
                with tab2:
                    st.subheader(f"Analysis: {selected_sheet}")
                    
                    # Display basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", analysis.get('rows', 0))
                    with col2:
                        st.metric("Columns", analysis.get('columns', 0))
                    with col3:
                        missing_pct = analysis.get('missing_values', {}).get('percentage', 0)
                        st.metric("Missing Values", f"{missing_pct:.2f}%")
                    
                    # Display column information
                    st.subheader("Column Information")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Numerical Columns:")
                        num_cols = analysis.get('numerical_columns', [])
                        if num_cols:
                            st.write(", ".join(num_cols))
                        else:
                            st.write("No numerical columns found")
                    
                    with col2:
                        st.write("Categorical Columns:")
                        cat_cols = analysis.get('categorical_columns', [])
                        if cat_cols:
                            st.write(", ".join(cat_cols))
                        else:
                            st.write("No categorical columns found")
                    
                    # Display statistics for numerical columns
                    if 'statistics' in analysis:
                        st.subheader("Numerical Statistics")
                        
                        # Create a dataframe with the statistics
                        stats_data = []
                        for col in analysis.get('numerical_columns', []):
                            if col in analysis['statistics']['mean']:
                                stats_data.append({
                                    'Column': col,
                                    'Mean': analysis['statistics']['mean'].get(col, np.nan),
                                    'Std Dev': analysis['statistics']['std'].get(col, np.nan),
                                    'Min': analysis['statistics']['min'].get(col, np.nan),
                                    '25%': analysis['statistics']['25%'].get(col, np.nan),
                                    'Median': analysis['statistics']['50%'].get(col, np.nan),
                                    '75%': analysis['statistics']['75%'].get(col, np.nan),
                                    'Max': analysis['statistics']['max'].get(col, np.nan)
                                })
                        
                        if stats_data:
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df)
                
                with tab3:
                    st.subheader(f"Charts: {selected_sheet}")
                    
                    if charts:
                        for i, (chart_type, fig) in enumerate(charts):
                            st.subheader(f"{chart_type.capitalize()} Chart {i+1}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        if sheet_meta.get('has_charts', False):
                            st.info("Charts were detected in this sheet but could not be automatically recreated.")
                        else:
                            st.info("No charts were detected in this sheet.")
                        
                        # Offer to create standard charts
                        st.subheader("Create Standard Charts")
                        
                        num_cols = analysis.get('numerical_columns', [])
                        cat_cols = analysis.get('categorical_columns', [])
                        
                        if num_cols and cat_cols:
                            chart_type = st.selectbox(
                                "Select chart type:",
                                ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Box Plot"]
                            )
                            
                            if chart_type == "Bar Chart":
                                x_col = st.selectbox("Select X-axis column (category):", cat_cols)
                                y_col = st.selectbox("Select Y-axis column (numeric):", num_cols)
                                
                                if st.button("Generate Bar Chart"):
                                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            elif chart_type == "Line Chart":
                                # Check if there are date columns to use as x-axis
                                date_cols = analysis.get('date_columns', [])
                                x_options = date_cols if date_cols else cat_cols
                                
                                x_col = st.selectbox("Select X-axis column:", x_options)
                                y_col = st.selectbox("Select Y-axis column:", num_cols, key="line_y")
                                
                                if st.button("Generate Line Chart"):
                                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            elif chart_type == "Scatter Plot":
                                x_col = st.selectbox("Select X-axis column:", num_cols, key="scatter_x")
                                y_col = st.selectbox("Select Y-axis column:", 
                                                    [c for c in num_cols if c != x_col] if len(num_cols) > 1 else num_cols,
                                                    key="scatter_y")
                                
                                color_by = None
                                if cat_cols:
                                    use_color = st.checkbox("Color by category")
                                    if use_color:
                                        color_by = st.selectbox("Select color column:", cat_cols)
                                
                                if st.button("Generate Scatter Plot"):
                                    fig = px.scatter(df, x=x_col, y=y_col, color=color_by,
                                                    title=f"{y_col} vs {x_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            elif chart_type == "Pie Chart":
                                name_col = st.selectbox("Select category column:", cat_cols)
                                value_col = st.selectbox("Select value column:", num_cols, key="pie_y")
                                
                                if st.button("Generate Pie Chart"):
                                    # For pie charts, limit to top N categories if there are too many
                                    if df[name_col].nunique() > 10:
                                        # Group by category and sum values
                                        pie_data = df.groupby(name_col)[value_col].sum().reset_index()
                                        # Sort by value descending
                                        pie_data = pie_data.sort_values(value_col, ascending=False)
                                        # Take top 9 and group others
                                        top_data = pie_data.head(9)
                                        other_sum = pie_data.iloc[9:][value_col].sum()
                                        other_row = pd.DataFrame({name_col: ['Other'], value_col: [other_sum]})
                                        pie_data = pd.concat([top_data, other_row])
                                        
                                        fig = px.pie(pie_data, names=name_col, values=value_col, 
                                                    title=f"{value_col} by {name_col}")
                                    else:
                                        fig = px.pie(df, names=name_col, values=value_col, 
                                                    title=f"{value_col} by {name_col}")
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            elif chart_type == "Box Plot":
                                y_col = st.selectbox("Select numeric column:", num_cols, key="box_y")
                                x_col = st.selectbox("Select grouping column:", cat_cols, key="box_x")
                                
                                if st.button("Generate Box Plot"):
                                    fig = px.box(df, x=x_col, y=y_col, title=f"Distribution of {y_col} by {x_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            if not num_cols:
                                st.warning("No numerical columns found. Charts require numerical data.")
                            if not cat_cols:
                                st.warning("No categorical columns found. Some charts require categorical data.")
                
                with tab4:
                    st.subheader(f"Tables: {selected_sheet}")
                    
                    if tables:
                        for i, table in enumerate(tables):
                            st.subheader(f"Table {i+1}")
                            st.dataframe(table)
                    else:
                        st.info("No tables were detected in this sheet.")
                        
                        # Offer to create pivot tables
                        if len(analysis.get('numerical_columns', [])) > 0 and len(analysis.get('categorical_columns', [])) > 0:
                            st.subheader("Create Pivot Table")
                            
                            num_cols = analysis.get('numerical_columns', [])
                            cat_cols = analysis.get('categorical_columns', [])
                            
                            rows = st.selectbox("Select row dimension:", cat_cols)
                            cols = st.selectbox("Select column dimension:", 
                                              [c for c in cat_cols if c != rows] if len(cat_cols) > 1 else cat_cols,
                                              key="pivot_cols")
                            values = st.selectbox("Select values to aggregate:", num_cols)
                            
                            agg_func = st.selectbox("Select aggregation function:", 
                                                  ["Sum", "Mean", "Count", "Min", "Max"])
                            
                            if st.button("Generate Pivot Table"):
                                # Map string function name to actual function
                                agg_map = {
                                    "Sum": "sum",
                                    "Mean": "mean",
                                    "Count": "count",
                                    "Min": "min",
                                    "Max": "max"
                                }
                                
                                try:
                                    pivot = pd.pivot_table(df, values=values, index=rows, 
                                                        columns=cols, aggfunc=agg_map[agg_func])
                                    st.dataframe(pivot.style.format(precision=2))
                                    
                                    # Offer download of pivot table
                                    pivot_csv = pivot.to_csv()
                                    st.download_button(
                                        "Download Pivot Table",
                                        pivot_csv,
                                        "pivot_table.csv",
                                        "text/csv",
                                        key="download_pivot"
                                    )
                                except Exception as e:
                                    st.error(f"Error generating pivot table: {str(e)}")
            
            # Export options
            st.subheader("Export Analysis")
            
            if st.button("Export Complete Analysis to Excel"):
                with st.spinner("Generating Excel report..."):
                    excel_bytes = export_analysis_to_excel(
                        workbook_data,
                        st.session_state.analysis_results,
                        st.session_state.charts
                    )
                    
                    b64 = base64.b64encode(excel_bytes).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="excel_analysis.xlsx">Download Excel Analysis</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error analyzing Excel file: {str(e)}")

if __name__ == "__main__":
    display_excel_analyzer()