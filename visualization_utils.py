import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def apply_excel_theme(fig):
    """
    Apply Excel-like theme to any plotly figure

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to style

    Returns:
    --------
    plotly.graph_objects.Figure
        The styled figure
    """
    # Excel-like colors
    excel_colors = px.colors.qualitative.Plotly

    # Excel-like grid and layout
    fig.update_layout(
        font=dict(
            family="Times New Roman, Times, serif",
            size=12,
            color="#000000"
        ),
        title_font=dict(
            family="Times New Roman, Times, serif",
            size=16,
            color="#000000"
        ),
        legend=dict(
            font=dict(
                family="Times New Roman, Times, serif",
                size=10,
                color="#000000"
            ),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(211, 211, 211, 1)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=True,
            zerolinecolor='rgba(211, 211, 211, 1)',
            zerolinewidth=1,
            showline=True,
            linecolor='rgba(211, 211, 211, 1)',
            linewidth=1,
            title=dict(font=dict(family="Times New Roman, Times, serif")),
            tickfont=dict(family="Times New Roman, Times, serif"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=True,
            zerolinecolor='rgba(211, 211, 211, 1)',
            zerolinewidth=1,
            showline=True,
            linecolor='rgba(211, 211, 211, 1)',
            linewidth=1,
            title=dict(font=dict(family="Times New Roman, Times, serif")),
            tickfont=dict(family="Times New Roman, Times, serif"),
        ),
    )

    # Update marker colors using update_traces which is safer than direct property access
    if hasattr(fig, 'data') and fig.data:
        # Handle different trace types with a consistent approach
        for i, trace in enumerate(fig.data):
            color = excel_colors[i % len(excel_colors)]

            # Use update traces for a specific trace (safer than direct property access)
            # This handles both bar charts and scatter plots in a consistent way
            if trace.type == 'bar':
                fig.update_traces(
                    marker=dict(
                        color=color,
                        line=dict(color='white', width=1)
                    ),
                    selector=dict(type='bar', name=trace.name)
                )
            elif trace.type == 'scatter':
                # Handle scatter plots (lines, markers, or both)
                fig.update_traces(
                    line=dict(color=color, width=2),
                    marker=dict(color=color, line=dict(
                        color='white', width=1)),
                    selector=dict(type='scatter', name=trace.name)
                )
            elif trace.type == 'pie':
                # For pie charts, ensure Excel-like styling
                fig.update_traces(
                    marker=dict(line=dict(color='white', width=1)),
                    textfont=dict(family="Times New Roman", size=10),
                    selector=dict(type='pie')
                )

            return fig


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
            title=f"Histogram of {column}",
            template="plotly_white"  # Clean white template as base
        )

        # Add a KDE line
        hist_data = [df[column].dropna()]
        group_labels = [column]

        try:
            # This can fail for small datasets or those with low variance
            kde_fig = ff.create_distplot(
                hist_data, group_labels, show_hist=False, show_rug=False)
            kde_trace = kde_fig.data[0]
            fig.add_trace(kde_trace)
        except Exception as e:
            # Skip KDE if it fails
            pass

        # Add mean line
        mean_val = df[column].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#FF0000",  # Excel red
                      annotation_text=f"Mean: {mean_val:.2f}",
                      annotation_position="top right",
                      annotation_font=dict(family="Times New Roman"))

        # Add median line
        median_val = df[column].median()
        fig.add_vline(x=median_val, line_dash="dash", line_color="#009900",  # Excel green
                      annotation_text=f"Median: {median_val:.2f}",
                      annotation_position="top left",
                      annotation_font=dict(family="Times New Roman"))

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
            labels={column: column, 'Count': 'Frequency'},
            template="plotly_white",  # Clean white template as base
            color_discrete_sequence=px.colors.qualitative.Plotly  # Excel-like colors
        )

        # Add data labels on top of bars (Excel-like)
        fig.update_traces(
            texttemplate='%{y}',
            textposition='outside'
        )

        # Update layout for categorical histogram
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Frequency",
            xaxis={'categoryorder': 'total descending'}  # Sort by frequency
        )

    # Apply Excel theme to the figure
    fig = apply_excel_theme(fig)

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
                points="all",  # Show all points
                template="plotly_white",  # Clean white template as base
                color_discrete_sequence=px.colors.qualitative.Plotly  # Excel-like colors
            )
        else:
            title = f"Box Plot of {column}"
            fig = px.box(
                df,
                y=column,
                title=title,
                points="all",  # Show all points
                template="plotly_white",  # Clean white template as base
                color_discrete_sequence=px.colors.qualitative.Plotly  # Excel-like colors
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
            counts = df.groupby(
                [group_by, column]).size().reset_index(name='Count')
            title = f"Value Counts of {column} by {group_by}"
            fig = px.bar(
                counts,
                x=column,
                y='Count',
                color=group_by,
                title=title,
                barmode='group',
                template="plotly_white",  # Clean white template as base
                color_discrete_sequence=px.colors.qualitative.Plotly  # Excel-like colors
            )

            # Add data labels on top of bars (Excel-like)
            fig.update_traces(
                texttemplate='%{y}',
                textposition='outside'
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
                title=title,
                template="plotly_white",  # Clean white template as base
                color_discrete_sequence=px.colors.qualitative.Plotly  # Excel-like colors
            )

            # Add data labels on top of bars (Excel-like)
            fig.update_traces(
                texttemplate='%{y}',
                textposition='outside'
            )

            # Update layout
            fig.update_layout(
                xaxis_title=column,
                yaxis_title='Count',
                # Sort by frequency
                xaxis={'categoryorder': 'total descending'}
            )

    # Apply Excel theme to the figure
    fig = apply_excel_theme(fig)

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
    both_numeric = pd.api.types.is_numeric_dtype(
        df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])

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
            trendline="ols" if both_numeric else None,
            template="plotly_white",  # Clean white template as base
            color_discrete_sequence=px.colors.qualitative.Plotly  # Excel-like colors
        )
    else:
        title = f"Scatter Plot of {y_col} vs {x_col}"
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title,
            # Only add trendline if both columns are numeric
            trendline="ols" if both_numeric else None,
            template="plotly_white",  # Clean white template as base
            color_discrete_sequence=px.colors.qualitative.Plotly  # Excel-like colors
        )

    # Add correlation coefficient to the title only if both columns are numeric
    if both_numeric:
        try:
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            fig.update_layout(
                title=f"{title} (r = {corr:.3f})"
            )

            # Add correlation text in Excel style
            fig.add_annotation(
                x=0.02, y=0.95,
                xref="paper", yref="paper",
                text=f"r = {corr:.3f}",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                font=dict(family="Times New Roman", size=12, color="black")
            )

            # Format trendline to match Excel style - safer implementation
            try:
                # Check each trace safely
                for trace in fig.data:
                    # Check specifically for OLS trendlines (usually named as 'Trendline' or have mode='lines')
                    if hasattr(trace, 'name') and 'Trendline' in trace.name:
                        if hasattr(trace, 'line'):
                            # Set Excel-like trendline properties
                            trace.line.update(
                                color='rgba(255, 0, 0, 0.7)',
                                width=2,
                                dash='dash'
                            )
            except Exception:
                # Silently continue if trace property access fails
                pass
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

    # Apply Excel theme to the figure
    fig = apply_excel_theme(fig)

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

            # Create heatmap with Excel-like styling
            fig = px.imshow(
                corr_matrix,
                text_auto=True,  # Display correlation values on cells
                # Red-Blue color scale (Excel-like)
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title=f"{corr_method.capitalize()} Correlation Heatmap",
                template="plotly_white"  # Clean white template as base
            )

            # Update cell text formatting for Excel style
            fig.update_traces(
                texttemplate='%{z:.3f}',  # Format to 3 decimal places
                textfont=dict(family="Times New Roman", size=12, color="black")
            )

            # Update layout for Excel style
            fig.update_layout(
                width=700,
                height=700,
                coloraxis_colorbar=dict(
                    title="Correlation",
                    titlefont=dict(family="Times New Roman"),
                    tickfont=dict(family="Times New Roman")
                )
            )

            # Apply Excel theme
            fig = apply_excel_theme(fig)
            return fig
        else:
            # For non-numeric columns or mixed data, create a contingency-based visual
            if len(columns) == 2:
                # For two columns, create a heatmap of counts
                # Create contingency table
                cont_table = pd.crosstab(df[columns[0]], df[columns[1]])

                # Create heatmap with Excel-like styling
                fig = px.imshow(
                    cont_table,
                    text_auto=True,  # Display count values on cells
                    color_continuous_scale='Blues',  # Blue color scale
                    title=f"Contingency Table: {columns[0]} vs {columns[1]}",
                    template="plotly_white"  # Clean white template as base
                )

                # Update cell text formatting for Excel style
                fig.update_traces(
                    textfont=dict(family="Times New Roman",
                                  size=12, color="black")
                )

                # Update layout for Excel style
                fig.update_layout(
                    width=700,
                    height=700,
                    xaxis_title=columns[1],
                    yaxis_title=columns[0],
                    coloraxis_colorbar=dict(
                        title="Count",
                        titlefont=dict(family="Times New Roman"),
                        tickfont=dict(family="Times New Roman")
                    )
                )

                # Apply Excel theme
                fig = apply_excel_theme(fig)
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
                    font=dict(family="Times New Roman", size=14)
                )

                fig.update_layout(
                    title="Correlation Analysis Not Available",
                    width=700,
                    height=700,
                    title_font=dict(family="Times New Roman")
                )

                # Apply Excel theme
                fig = apply_excel_theme(fig)
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
            font=dict(family="Times New Roman", size=14, color="red")
        )

        fig.update_layout(
            title="Correlation Analysis Error",
            width=700,
            height=700,
            title_font=dict(family="Times New Roman")
        )

        # Apply Excel theme
        fig = apply_excel_theme(fig)
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

            # Create the distplot with Excel-like styling
            fig = ff.create_distplot(
                hist_data,
                group_labels,
                bin_size=(data.max() - data.min()) / 20,
                curve_type='normal',
                show_rug=True,
                colors=[px.colors.qualitative.Plotly[0]]  # Excel blue
            )

            # Update layout with Excel styling
            fig.update_layout(
                title=f"Distribution Plot of {column}",
                xaxis_title=column,
                yaxis_title="Density",
                showlegend=True,
                template="plotly_white"  # Clean white template as base
            )

            # Add descriptive statistics with Excel-like formatting
            stats_text = (
                f"Mean: {data.mean():.3f}<br>"
                f"Median: {data.median():.3f}<br>"
                f"Std Dev: {data.std():.3f}<br>"
                f"Min: {data.min():.3f}<br>"
                f"Max: {data.max():.3f}"
            )

            # Add stats box in Excel style
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                align="right",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(180, 180, 180, 1)",
                borderwidth=1,
                font=dict(family="Times New Roman", size=12, color="black")
            )

            # Apply Excel theme to the figure
            fig = apply_excel_theme(fig)

            return fig

        except Exception as e:
            # Fallback to histogram if distplot fails
            return create_histogram(df, column)
    else:
        # For non-numeric data, create a value counts bar chart with Excel styling
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
            title=title,
            template="plotly_white",  # Clean white template as base
            color_discrete_sequence=px.colors.qualitative.Plotly  # Excel-like colors
        )

        # Add data labels on top of bars (Excel-like)
        fig.update_traces(
            texttemplate='%{y}',
            textposition='outside',
            textfont=dict(family="Times New Roman", size=10)
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

        # Add stats box in Excel style
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(180, 180, 180, 1)",
            borderwidth=1,
            font=dict(family="Times New Roman", size=12, color="black")
        )

        # Update layout for Excel-like style
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            xaxis={'categoryorder': 'total descending'}
        )

        # Apply Excel theme to the figure
        fig = apply_excel_theme(fig)

        return fig


def create_frequency_analysis(df, column, include_percentage=True, sort_by='count', chart_type='pie'):
    """
    Create frequency analysis for categorical data with counts and percentages

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    column : str
        The categorical column to analyze
    include_percentage : bool
        Whether to include percentage in the results
    sort_by : str
        How to sort the results ('count' or 'alphabetical')
    chart_type : str
        Type of chart to create ('pie', 'bar', or 'both')

    Returns:
    --------
    pd.DataFrame, plotly.graph_objects.Figure or list
        DataFrame with frequency analysis and visualization(s)
    """
    # Get value counts
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'Count']

    # Calculate percentages
    if include_percentage:
        total = value_counts['Count'].sum()
        value_counts['Percentage'] = (
            value_counts['Count'] / total * 100).round(1).astype(str) + '%'

    # Sort the results as requested
    if sort_by == 'count':
        value_counts = value_counts.sort_values('Count', ascending=False)
    elif sort_by == 'alphabetical':
        value_counts = value_counts.sort_values(column)

    # Create figures based on chart type
    figures = []

    if chart_type in ['pie', 'both']:
        # Create a percentage pie chart with Excel-like styling
        fig_percent = px.pie(
            value_counts,
            values='Count',
            names=column,
            title=f"Percentage Distribution of {column}",
            hover_data=['Count', 'Percentage'],
            color_discrete_sequence=px.colors.qualitative.Plotly,  # Excel-like color scheme
        )

        # Create custom hovertext to show counts and percentages
        hover_text = [
            f"{row[column]}: {row['Count']} ({row['Percentage']})" for _, row in value_counts.iterrows()]

        fig_percent.update_traces(
            textinfo='none',  # Don't show any text on the pie slices as requested
            showlegend=True,
            pull=[0.02] * len(value_counts),  # Slight pull for all slices
            # White borders between slices
            marker=dict(line=dict(color='white', width=1)),
            hovertext=hover_text,  # Add custom hover text with count information
            hoverinfo='text'
        )

        # Apply Excel styling
        fig_percent = apply_excel_theme(fig_percent)
        figures.append(fig_percent)

    if chart_type in ['pie', 'both']:
        # Create a count pie chart with Excel-like styling
        fig_counts = px.pie(
            value_counts,
            values='Count',
            names=column,
            title=f"Count Distribution of {column}",
            hover_data=['Count', 'Percentage'],
            color_discrete_sequence=px.colors.qualitative.Plotly,  # Excel-like color scheme
        )

        # Create custom hovertext showing raw counts
        hover_text = [f"{row[column]}: {row['Count']}" for _,
                      row in value_counts.iterrows()]

        fig_counts.update_traces(
            textinfo='none',  # Don't show any text on the pie slices as requested
            showlegend=True,
            pull=[0.02] * len(value_counts),  # Slight pull for all slices
            # White borders between slices
            marker=dict(line=dict(color='white', width=1)),
            hovertext=hover_text,  # Add custom hover text
            hoverinfo='text'
        )

        # Apply Excel styling
        fig_counts = apply_excel_theme(fig_counts)
        figures.append(fig_counts)

    # If we have bar charts, create and add them
    if chart_type in ['bar', 'both']:
        # Create a bar chart for counts
        fig_bar = px.bar(
            value_counts,
            x=column,
            y='Count',
            title=f"Frequency Count of {column}",
            color=column,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            template="plotly_white"
        )

        # Apply Excel-like styling
        fig_bar = apply_excel_theme(fig_bar)
        fig_bar.update_traces(
            textfont=dict(family="Times New Roman, Times, serif", size=10),
            texttemplate='%{y}',
            textposition='outside'
        )
        figures.append(fig_bar)

    # Apply general Excel-like styling to all figures
    for i, fig in enumerate(figures):
        fig.update_layout(
            legend=dict(
                bgcolor='rgba(240, 240, 240, 0.8)',
                bordercolor='rgba(180, 180, 180, 1)',
                borderwidth=1,
                title=dict(text=column, font=dict(
                    family="Times New Roman, Times, serif")),
                font=dict(family="Times New Roman, Times, serif"),
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1.1
            ),
            title_font=dict(family="Times New Roman, Times, serif", size=14),
            font=dict(family="Times New Roman, Times, serif"),
            height=500,
            width=700
        )

    # Return the analysis dataframe and figure(s)
    if len(figures) == 1:
        return value_counts, figures[0]
    else:
        return value_counts, figures


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

            # Create Q-Q plot using a DataFrame approach which is safer
            # Create a temporary dataframe for the plot
            qq_df = pd.DataFrame({
                'theoretical_quantiles': np.sort(np.random.normal(0, 1, len(data))),
                'sample_quantiles': np.sort((data - data.mean()) / data.std())
            })

            # Use px.scatter with the dataframe
            fig = px.scatter(
                qq_df,
                x='theoretical_quantiles',
                y='sample_quantiles',
                title=f"Q-Q Plot of {column}",
                template="plotly_white"  # Use clean white template as base
            )

            # Add the identity line using the dataframe min/max
            min_val = min(qq_df['theoretical_quantiles'].min(),
                          qq_df['sample_quantiles'].min())
            max_val = max(qq_df['theoretical_quantiles'].max(),
                          qq_df['sample_quantiles'].max())

            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Normal',
                    line=dict(color='red', width=2, dash='dash')
                )
            )

            # Update layout with Excel-like styling
            fig.update_layout(
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                showlegend=True,
                font=dict(family="Times New Roman, Times, serif"),
                title_font=dict(
                    family="Times New Roman, Times, serif", size=14),
                legend=dict(font=dict(family="Times New Roman, Times, serif"))
            )

            # Apply Excel theme to ensure consistent styling across all visualizations
            fig = apply_excel_theme(fig)

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
                    borderwidth=1,
                    font=dict(family="Times New Roman, Times, serif", size=10)
                )
            except:
                # If normality test fails, continue without it
                pass

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


def generate_dynamic_plotly_chart(df, chart_config, chart_type="bar"):
    x_axis = chart_config.get('x_axis')
    y_axes = chart_config.get('y_axes', [])  # Default to empty list
    color_by = chart_config.get('color')
    size_by = chart_config.get('size')
    facet_row = chart_config.get('facet_row')
    facet_col = chart_config.get('facet_col')
    # You might add an 'agg_func' to chart_config from a selectbox in app.py

    if not x_axis or not y_axes:
        fig = go.Figure()
        fig.add_annotation(text="Configure X and Y axes", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Chart Configuration Needed")
        return fig

    y_to_plot = y_axes if isinstance(y_axes, list) else [y_axes]
    if not y_to_plot:
        fig = go.Figure()
        fig.add_annotation(text="Configure Y axis", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Y-Axis Needed")
        return fig

    title = f"{chart_type.capitalize()} of {', '.join(y_to_plot)} by {x_axis}"
    if color_by:
        title += f" (Colored by {color_by})"
    if facet_row:
        title += f" (Faceted by {facet_row})"
    if facet_col:
        title += f" (Faceted by {facet_col})"

    plot_kwargs = {
        "x": x_axis,
        # Some charts handle list of y better
        "y": y_to_plot[0] if len(y_to_plot) == 1 and chart_type not in ["bar", "line"] else y_to_plot,
        "color": color_by,
        "size": size_by if chart_type == "scatter" else None,
        "facet_row": facet_row,
        "facet_col": facet_col,
        "title": title
    }
    # Remove None values from kwargs
    plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}

    try:
        if chart_type == "bar":
            # For bar charts, if multiple y_axes, Plotly Express handles them well with barmode='group' (default)
            fig = px.bar(df, **plot_kwargs)
        elif chart_type == "line":
            fig = px.line(df, **plot_kwargs)
        elif chart_type == "scatter":
            # Ensure 'y' is not a list for scatter if only one y is intended, or handle multiple y's appropriately
            if isinstance(plot_kwargs.get("y"), list) and len(plot_kwargs["y"]) > 1:
                # Plotly scatter can take a list for y, creating multiple traces if a color/symbol dimension is not also a list of same length.
                # Or, you might want to plot only the first one by default for simplicity, or let user choose.
                pass  # px.scatter can often handle this, or you might iterate to create traces
            fig = px.scatter(df, **plot_kwargs)
        elif chart_type == "pie":
            if y_to_plot and len(y_to_plot) == 1:
                # Pie usually takes one value column, names from x_axis or another category
                fig = px.pie(df, names=x_axis,
                             values=y_to_plot[0], color=color_by, title=title)
            else:
                raise ValueError(
                    "Pie charts require one column for values and one for names/categories.")
        elif chart_type == "histogram":
            # Histogram usually on one of y_axes, x_axis can be used for color
            fig = px.histogram(df, x=y_to_plot[0] if y_to_plot else x_axis,
                               color=x_axis if y_to_plot else color_by, title=title)  # adjust logic
        elif chart_type == "box":
            # Adjust x/y for box
            fig = px.box(
                df, x=x_axis, y=y_to_plot[0] if y_to_plot else None, color=color_by, title=title)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        fig.update_layout(title_x=0.5)
        return fig
    except Exception as e:
        # Better error display on the chart itself
        error_fig = go.Figure()
        error_fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False, align="center", width=500)
        error_fig.update_layout(title="Chart Generation Error")
        return error_fig
