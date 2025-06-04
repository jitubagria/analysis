import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import io
import base64
from datetime import datetime
from data_utils import get_descriptive_stats
from visualization_utils import create_histogram, create_boxplot, create_correlation_heatmap

def generate_report(df, filename="dataset.csv", include_summary=True, include_stats=True, 
                   include_viz=True, include_corr=True):
    """
    Generate an HTML report from the data
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    filename : str
        Name of the original file
    include_summary : bool
        Whether to include data summary
    include_stats : bool
        Whether to include descriptive statistics
    include_viz : bool
        Whether to include visualizations
    include_corr : bool
        Whether to include correlation analysis
        
    Returns:
    --------
    str
        HTML report as a string
    """
    # Get basic info about the dataframe
    num_rows, num_cols = df.shape
    num_missing = df.isnull().sum().sum()
    percent_missing = (num_missing / (num_rows * num_cols) * 100)
    
    # Numerical and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Start building HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Statistical Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .container {{
                margin: 20px 0;
                overflow-x: auto;
            }}
            .missing-summary {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }}
            .missing-card {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                flex: 1;
                min-width: 200px;
            }}
            .viz-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }}
            .viz-item {{
                flex: 1;
                min-width: 300px;
                margin-bottom: 20px;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Statistical Analysis Report</h1>
        <p><strong>Dataset:</strong> {filename}</p>
        <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Data Summary section
    if include_summary:
        html += f"""
        <h2>1. Data Summary</h2>
        <div class="container">
            <h3>Dataset Overview</h3>
            <div class="missing-summary">
                <div class="missing-card">
                    <h4>Rows</h4>
                    <p>{num_rows}</p>
                </div>
                <div class="missing-card">
                    <h4>Columns</h4>
                    <p>{num_cols}</p>
                </div>
                <div class="missing-card">
                    <h4>Missing Values</h4>
                    <p>{num_missing} ({percent_missing:.2f}%)</p>
                </div>
                <div class="missing-card">
                    <h4>Numerical Columns</h4>
                    <p>{len(num_cols)}</p>
                </div>
                <div class="missing-card">
                    <h4>Categorical Columns</h4>
                    <p>{len(cat_cols)}</p>
                </div>
            </div>
        </div>
        
        <div class="container">
            <h3>Data Preview</h3>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        {" ".join([f"<th>{col}</th>" for col in df.columns])}
                    </tr>
                </thead>
                <tbody>
                    {" ".join([f"<tr><td>{i}</td>{' '.join([f'<td>{str(row[col])[:50]}</td>' for col in df.columns])}</tr>" for i, row in df.head(10).iterrows()])}
                </tbody>
            </table>
            <p><em>Showing 10 of {num_rows} rows</em></p>
        </div>
        
        <div class="container">
            <h3>Column Information</h3>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Data Type</th>
                        <th>Missing Values</th>
                        <th>Missing %</th>
                        <th>Unique Values</th>
                    </tr>
                </thead>
                <tbody>
                    {" ".join([f"<tr><td>{col}</td><td>{df[col].dtype}</td><td>{df[col].isnull().sum()}</td><td>{df[col].isnull().mean()*100:.2f}%</td><td>{df[col].nunique()}</td></tr>" for col in df.columns])}
                </tbody>
            </table>
        </div>
        """
    
    # Descriptive Statistics section
    if include_stats and num_cols:
        # Generate statistics for all numerical columns
        stats_df = get_descriptive_stats(df, num_cols)
        
        html += f"""
        <h2>2. Descriptive Statistics</h2>
        <div class="container">
            <h3>Numerical Columns</h3>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        {" ".join([f"<th>{col}</th>" for col in stats_df.columns])}
                    </tr>
                </thead>
                <tbody>
                    {" ".join([f"<tr><td>{idx}</td>{' '.join([f'<td>{stats_df.loc[idx, col]:.4f}</td>' if isinstance(stats_df.loc[idx, col], (int, float)) and not pd.isna(stats_df.loc[idx, col]) else f'<td>{stats_df.loc[idx, col]}</td>' for col in stats_df.columns])}</tr>" for idx in stats_df.index])}
                </tbody>
            </table>
        </div>
        """
        
        # Add frequency tables for categorical columns if available
        if cat_cols:
            html += f"""
            <div class="container">
                <h3>Categorical Columns</h3>
            """
            
            for col in cat_cols[:5]:  # Limit to 5 categorical columns to avoid huge reports
                freq_table = df[col].value_counts().reset_index()
                freq_table.columns = [col, 'Count']
                freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
                freq_table = freq_table.head(10)  # Show only top 10 values
                
                html += f"""
                <h4>{col}</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Value</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {" ".join([f"<tr><td>{row[col]}</td><td>{row['Count']}</td><td>{row['Percentage']:.2f}%</td></tr>" for _, row in freq_table.iterrows()])}
                    </tbody>
                </table>
                <p><em>Showing top 10 values</em></p>
                """
            
            html += """
            </div>
            """
    
    # Visualizations section
    if include_viz and num_cols:
        html += """
        <h2>3. Data Visualizations</h2>
        """
        
        # Add histograms for numerical columns
        if num_cols:
            html += """
            <div class="container">
                <h3>Distributions</h3>
                <div class="viz-container">
            """
            
            # Create histograms for up to 6 numerical columns
            for col in num_cols[:6]:
                fig = create_histogram(df, col)
                plot_div = fig.to_html(full_html=False, include_plotlyjs=False)
                
                html += f"""
                <div class="viz-item">
                    {plot_div}
                </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        # Add box plots if there are categorical columns
        if cat_cols and num_cols:
            html += """
            <div class="container">
                <h3>Comparative Analysis</h3>
                <div class="viz-container">
            """
            
            # Create a box plot for the first numerical column grouped by the first categorical column
            # (limited to save space)
            for num_col in num_cols[:3]:
                for cat_col in cat_cols[:2]:
                    if df[cat_col].nunique() <= 10:  # Only if the categorical column has a reasonable number of values
                        fig = create_boxplot(df, num_col, cat_col)
                        plot_div = fig.to_html(full_html=False, include_plotlyjs=False)
                        
                        html += f"""
                        <div class="viz-item">
                            {plot_div}
                        </div>
                        """
            
            html += """
                </div>
            </div>
            """
    
    # Correlation Analysis section
    if include_corr and len(num_cols) >= 2:
        html += """
        <h2>4. Correlation Analysis</h2>
        <div class="container">
        """
        
        # Create a correlation heatmap
        # Limit to 10 columns to avoid creating a too large heatmap
        corr_cols = num_cols[:10]
        fig = create_correlation_heatmap(df, corr_cols, 'pearson')
        plot_div = fig.to_html(full_html=False, include_plotlyjs=False)
        
        html += f"""
            <h3>Correlation Heatmap</h3>
            {plot_div}
            
            <h3>Correlation Matrix</h3>
            <table>
                <thead>
                    <tr>
                        <th>Variable</th>
                        {" ".join([f"<th>{col}</th>" for col in corr_cols])}
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add correlation matrix
        corr_matrix = df[corr_cols].corr()
        for idx in corr_matrix.index:
            html += f"""
                <tr>
                    <td>{idx}</td>
                    {" ".join([f'<td>{corr_matrix.loc[idx, col]:.4f}</td>' for col in corr_matrix.columns])}
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
    
    # Footer
    html += """
        <div class="footer">
            <p>Generated by Statistical Analysis Portal</p>
        </div>
        <script>
            window.onload = function() {
                // Resize Plotly charts to fit container width
                window.dispatchEvent(new Event('resize'));
            };
        </script>
    </body>
    </html>
    """
    
    return html

def encode_plot_to_base64(fig):
    """
    Convert a plotly figure to a base64 encoded image
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The plotly figure to convert
        
    Returns:
    --------
    str
        Base64 encoded image
    """
    img_bytes = fig.to_image(format="png")
    encoded = base64.b64encode(img_bytes).decode('ascii')
    return f"data:image/png;base64,{encoded}"
