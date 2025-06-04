import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_histogram(df, column, bins=20):
    """
    Create a histogram for any column type
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    column : str
        The column to visualize
    bins : int
        Number of bins for the histogram
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The histogram figure
    """
    # Check if column is numerical or categorical
    if pd.api.types.is_numeric_dtype(df[column]):
        # For numerical columns, create a standard histogram with statistics
        fig = px.histogram(
            df, 
            x=column,
            nbins=bins,
            marginal="box",  # Add a box plot on the margin
            histnorm="probability density",  # Normalize to show density instead of counts
            title=f"Histogram of {column}"
        )
        
        # Add a KDE line
        hist_data = [df[column].dropna()]
        group_labels = [column]
        
        try:
            # This can fail for small datasets or those with low variance
            kde_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
            kde_trace = kde_fig.data[0]
            fig.add_trace(kde_trace)
        except Exception as e:
            # Skip KDE if it fails
            pass
        
        # Add mean line
        mean_val = df[column].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                    annotation_text=f"Mean: {mean_val:.2f}", 
                    annotation_position="top right")
        
        # Add median line
        median_val = df[column].median()
        fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                    annotation_text=f"Median: {median_val:.2f}", 
                    annotation_position="top left")
        
        # Update layout
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Density",
            showlegend=True
        )
    else:
        # For categorical or non-numeric columns, create a count-based histogram
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'Count']
        
        # If too many categories, limit display
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            subtitle = " (showing top 20 categories)"
        else:
            subtitle = ""
        
        fig = px.bar(
            value_counts, 
            x=column, 
            y='Count',
            title=f"Histogram of {column}{subtitle}",
            labels={column: column, 'Count': 'Frequency'}
        )
        
        # Update layout for categorical histogram
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Frequency",
            xaxis={'categoryorder':'total descending'}  # Sort by frequency
        )
    
    return fig

def create_boxplot(df, column, group_by=None):
    """
    Create a box plot for any column type, optionally grouped by another column
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    column : str
        The column to visualize
    group_by : str, optional
        The column to group by
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The box plot figure
    """
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(df[column]):
        # For numeric data, create a standard box plot
        if group_by:
            title = f"Box Plot of {column} by {group_by}"
            fig = px.box(
                df, 
                x=group_by, 
                y=column, 
                color=group_by,
                title=title,
                points="all"  # Show all points
            )
        else:
            title = f"Box Plot of {column}"
            fig = px.box(
                df, 
                y=column,
                title=title,
                points="all"  # Show all points
            )
        
        # Update layout
        fig.update_layout(
            yaxis_title=column,
            showlegend=True if group_by else False
        )
    else:
        # For non-numeric data, create a count-based visualization similar to a box plot
        if group_by:
            # Create a count-based grouped visualization
            counts = df.groupby([group_by, column]).size().reset_index(name='Count')
            title = f"Value Counts of {column} by {group_by}"
            fig = px.bar(
                counts,
                x=column,
                y='Count',
                color=group_by,
                title=title,
                barmode='group'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=column,
                yaxis_title='Count',
                showlegend=True
            )
        else:
            # Create a simple count-based visualization
            counts = df[column].value_counts().reset_index()
            counts.columns = [column, 'Count']
            title = f"Value Counts of {column}"
            
            # If too many categories, limit display
            if len(counts) > 20:
                counts = counts.head(20)
                title += " (showing top 20 values)"
            
            fig = px.bar(
                counts,
                x=column,
                y='Count',
                title=title
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=column,
                yaxis_title='Count',
                xaxis={'categoryorder':'total descending'}  # Sort by frequency
            )
    
    return fig

def create_scatterplot(df, x_col, y_col, color_by=None):
    """
    Create a scatter plot between any column types
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    x_col : str
        The column for the x-axis
    y_col : str
        The column for the y-axis
    color_by : str, optional
        The column to color points by
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The scatter plot figure
    """
    # Check if both columns are numeric for correlation calculation
    both_numeric = pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])
    
    # Base title and configuration
    if color_by:
        title = f"Scatter Plot of {y_col} vs {x_col} (colored by {color_by})"
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_by,
            title=title,
            # Only add trendline if both columns are numeric
            trendline="ols" if both_numeric else None
        )
    else:
        title = f"Scatter Plot of {y_col} vs {x_col}"
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            title=title,
            # Only add trendline if both columns are numeric
            trendline="ols" if both_numeric else None
        )
    
    # Add correlation coefficient to the title only if both columns are numeric
    if both_numeric:
        try:
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            fig.update_layout(
                title=f"{title} (r = {corr:.3f})"
            )
        except Exception as e:
            # If correlation calculation fails, keep original title
            pass
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=True if color_by else False
    )
    
    # If one or both columns are categorical, use appropriate axis settings
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        fig.update_xaxes(type='category')
    
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        fig.update_yaxes(type='category')
    
    return fig

def create_correlation_heatmap(df, columns, corr_method='pearson'):
    """
    Create a correlation heatmap for selected columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    columns : list
        List of columns to include in the correlation analysis
    corr_method : str
        Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The heatmap figure
    """
    try:
        # Filter numeric columns if present in the selection
        numeric_columns = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
        
        # If we have numeric columns, calculate correlation
        if len(numeric_columns) >= 2:
            # Calculate correlation matrix
            corr_matrix = df[numeric_columns].corr(method=corr_method)
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,  # Display correlation values on cells
                color_continuous_scale='RdBu_r',  # Red-Blue color scale
                zmin=-1,
                zmax=1,
                title=f"{corr_method.capitalize()} Correlation Heatmap"
            )
            
            # Update layout
            fig.update_layout(
                width=700,
                height=700
            )
            
            return fig
        else:
            # For non-numeric columns or mixed data, create a contingency-based visual
            if len(columns) == 2:
                # For two columns, create a heatmap of counts
                # Create contingency table
                cont_table = pd.crosstab(df[columns[0]], df[columns[1]])
                
                # Create heatmap
                fig = px.imshow(
                    cont_table,
                    text_auto=True,  # Display count values on cells
                    color_continuous_scale='Blues',  # Blue color scale
                    title=f"Contingency Table: {columns[0]} vs {columns[1]}"
                )
                
                # Update layout
                fig.update_layout(
                    width=700,
                    height=700,
                    xaxis_title=columns[1],
                    yaxis_title=columns[0]
                )
                
                return fig
            else:
                # For multiple columns, create a simple placeholder with informative message
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Cannot create correlation heatmap for non-numeric columns.<br>Selected: {', '.join(columns)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14)
                )
                
                fig.update_layout(
                    title="Correlation Analysis Not Available",
                    width=700, 
                    height=700
                )
                
                return fig
    except Exception as e:
        # If all else fails, create an error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating correlation heatmap:<br>{str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        
        fig.update_layout(
            title="Correlation Analysis Error",
            width=700, 
            height=700
        )
        
        return fig

def create_distribution_plot(df, column):
    """
    Create a distribution plot for any column type
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    column : str
        The column to visualize
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The distribution plot figure
    """
    # Get the data
    data = df[column].dropna()
    
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(df[column]):
        # For numeric data, create a distribution plot
        try:
            # Create figure
            hist_data = [data]
            group_labels = [column]
            
            fig = ff.create_distplot(
                hist_data, 
                group_labels, 
                bin_size=(data.max() - data.min()) / 20,
                curve_type='normal',
                show_rug=True
            )
            
            # Update layout
            fig.update_layout(
                title=f"Distribution Plot of {column}",
                xaxis_title=column,
                yaxis_title="Density",
                showlegend=True
            )
            
            # Add descriptive statistics
            stats_text = (
                f"Mean: {data.mean():.3f}<br>"
                f"Median: {data.median():.3f}<br>"
                f"Std Dev: {data.std():.3f}<br>"
                f"Min: {data.min():.3f}<br>"
                f"Max: {data.max():.3f}"
            )
            
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                align="right",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            )
            
            return fig
        
        except Exception as e:
            # Fallback to histogram if distplot fails
            return create_histogram(df, column)
    else:
        # For non-numeric data, create a value counts bar chart
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'Count']
        
        # Sort by frequency, show top values if too many
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            title = f"Value Distribution of {column} (Top 20)"
        else:
            title = f"Value Distribution of {column}"
        
        fig = px.bar(
            value_counts, 
            x=column, 
            y='Count',
            title=title
        )
        
        # Add descriptive statistics
        total = len(data)
        top_value = value_counts.iloc[0][column]
        top_count = value_counts.iloc[0]['Count']
        unique_count = len(value_counts)
        
        stats_text = (
            f"Total Values: {total}<br>"
            f"Unique Values: {unique_count}<br>"
            f"Most Common: {top_value} ({top_count} occurrences)<br>"
            f"Most Common %: {(top_count/total*100):.1f}%"
        )
        
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            xaxis={'categoryorder':'total descending'}
        )
        
        return fig

def create_qq_plot(df, column):
    """
    Create a quantile-quantile (Q-Q) plot for a numerical column
    or an appropriate alternative for non-numeric data
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    column : str
        The column to visualize
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The Q-Q plot figure or alternative visualization
    """
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(df[column]):
        try:
            # Get the data
            data = df[column].dropna()
            
            # Create Q-Q plot
            fig = px.scatter(
                x=np.sort(np.random.normal(0, 1, len(data))),
                y=np.sort((data - data.mean()) / data.std()),
                title=f"Q-Q Plot of {column}"
            )
            
            # Add the identity line
            min_val = min(fig.data[0].x.min(), fig.data[0].y.min())
            max_val = max(fig.data[0].x.max(), fig.data[0].y.max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Normal',
                    line=dict(color='red', width=2, dash='dash')
                )
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                showlegend=True
            )
            
            # Add normality test results
            try:
                from scipy import stats
                k2, p_value = stats.normaltest(data)
                test_result = f"Normality Test: p-value = {p_value:.4f}"
                is_normal = "Data appears to be normally distributed" if p_value > 0.05 else "Data does not appear to be normally distributed"
                
                fig.add_annotation(
                    x=0.5,
                    y=0.02,
                    xref="paper",
                    yref="paper",
                    text=f"{test_result}<br>{is_normal}",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            except:
                # If normality test fails, continue without it
                pass
            
            return fig
        except Exception as e:
            # If QQ plot creation fails, return an informative message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating Q-Q plot for {column}:<br>{str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            
            fig.update_layout(
                title=f"Q-Q Plot Error for {column}",
                width=700, 
                height=500
            )
            
            return fig
    else:
        # For non-numeric data, show an informative message with cumulative distribution
        try:
            # Create alternative visualization using cumulative counts
            value_counts = df[column].value_counts().sort_values(ascending=False)
            cumulative = value_counts.cumsum() / value_counts.sum()
            
            # Create a dual-axis figure
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bar chart for counts
            fig.add_trace(
                go.Bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    name="Frequency",
                    marker_color="lightblue"
                ),
                secondary_y=False
            )
            
            # Add line chart for cumulative distribution
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index, 
                    y=cumulative.values,
                    name="Cumulative %",
                    marker_color="darkblue",
                    mode="lines+markers"
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title=f"Distribution Analysis of {column} (non-numeric)",
                xaxis_title=column,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Frequency", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
            
            # Add note about QQ plots
            fig.add_annotation(
                text="Note: Q-Q plots are only available for numeric data.<br>Showing distribution analysis instead.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.01,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            )
            
            return fig
        except Exception as e:
            # If alternative visualization fails, return a simple message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Q-Q Plot is not available for non-numeric column: {column}<br>Error creating alternative visualization: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            
            fig.update_layout(
                title="Q-Q Plot Not Available",
                width=700, 
                height=500
            )
            
            return fig
