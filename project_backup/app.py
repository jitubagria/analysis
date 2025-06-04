import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from data_utils import load_data, get_descriptive_stats, generate_age_groups, analyze_master_parameters
from stats_utils import perform_ttest, perform_anova, perform_correlation_analysis
from visualization_utils import (create_histogram, create_boxplot, create_scatterplot, 
                                create_correlation_heatmap, create_distribution_plot)
from report_generator import generate_report
from ml_utils import (analyze_data_structure, prepare_data_for_ml, train_model, 
                     perform_automated_analysis)
from drag_drop_ui import display_drag_drop_ui  # Using the new drag and drop UI
from drag_drop_config import display_config_tool  # Keep the old one as backup

# Set page configuration
st.set_page_config(
    page_title="Statistical Analysis Portal",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None
if 'visualization_type' not in st.session_state:
    st.session_state.visualization_type = None

def main():
    st.title("Statistical Analysis Portal 📊")
    
    # Create a horizontal navigation bar with buttons to jump to steps
    steps = [
        {"title": "1. Upload Excel", "step": 1}, 
        {"title": "2. Data Management", "step": 2}, 
        {"title": "3. Data Analysis", "step": 3}, 
        {"title": "4. Data Tables", "step": 4}, 
        {"title": "5. Visualizations", "step": 5}, 
        {"title": "6. Output", "step": 6}
    ]
    
    # Display horizontal step indicators with navigation buttons
    cols = st.columns(6)
    for i, (col, step) in enumerate(zip(cols, steps)):
        step_num = i + 1
        with col:
            if step_num == st.session_state.current_step:
                # Highlight current step with blue background
                st.markdown(f'<div style="background-color: #e6f2ff; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">{step["title"]}</div>', unsafe_allow_html=True)
            else:
                # Create a button for each step
                if st.button(step["title"], key=f"nav_{step['step']}"):
                    # Only allow jumping to steps if data is loaded (except for the first step)
                    if step["step"] == 1 or st.session_state.data is not None:
                        st.session_state.current_step = step["step"]
                        st.rerun()
    
    # Add a little space after the navigation
    st.write("")
    
    # Original welcome text
    st.write("""
    Welcome to the Statistical Analysis Portal! Follow the steps below to analyze your data:
    1. Upload your data file
    2. Manage and clean your data
    3. Select your desired analysis
    4. Review data tables
    5. Create visualizations
    6. Generate final output
    """)
    
    # Sidebar for navigation and steps
    with st.sidebar:
        st.header("Navigation")
        
        # Show current step indicator
        st.markdown(f"**Current Step: {st.session_state.current_step}/6**")
        
        # Step descriptions
        steps = {
            1: "Upload Excel",
            2: "Data Management",
            3: "Desired Data Analysis",
            4: "Tables of Data",
            5: "Graph/Chart Preparation",
            6: "Output"
        }
        
        for step_num, step_desc in steps.items():
            if step_num == st.session_state.current_step:
                st.markdown(f"**→ {step_num}. {step_desc}**")
            else:
                st.markdown(f"{step_num}. {step_desc}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.current_step > 1 and st.button("Previous Step"):
                st.session_state.current_step -= 1
                st.rerun()
        
        with col2:
            if st.session_state.current_step < 6 and st.session_state.data is not None and st.button("Next Step"):
                st.session_state.current_step += 1
                st.rerun()
                
        # Help information
        with st.expander("How to Use"):
            st.write("""
            **Excel Analysis Workflow:**
            
            1. **Upload Excel** - Upload your raw data (first sheet only)
            2. **Data Management** - Clean and prepare your data
            3. **Desired Analysis** - Choose the type of analysis to perform
            4. **Tables of Data** - View and customize data tables
            5. **Graph/Chart Preparation** - Create visualizations
            6. **Output** - Generate reports and export results
            
            This application replaces manual Excel analysis with an automated workflow.
            """)
    
    # Main content area based on current step
    if st.session_state.current_step == 1:
        # Step 1: Upload Excel
        st.header("1. Upload Excel")
        
        upload_col, sample_col = st.columns(2)
        
        with upload_col:
            st.subheader("Upload Your Data")
            # Simple file uploader
            uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls'])
            
            if uploaded_file is not None:
                try:
                    # Load data
                    df, filename = load_data(uploaded_file)
                    
                    # DEBUG: Check if the file already has Age Group columns
                    age_group_cols = [col for col in df.columns if 'age group' in col.lower() or 'Age group' in col]
                    if age_group_cols:
                        st.warning(f"Found existing Age Group columns in the file: {age_group_cols}")
                    
                    # Store the original file for multi-sheet analysis
                    if filename.endswith('.xlsx') or filename.endswith('.xls'):
                        st.session_state.excel_file = uploaded_file
                    
                    st.session_state.data = df
                    st.session_state.filename = filename
                    st.success(f"Successfully loaded: {filename}")
                    
                    # Show the complete raw data table with fixed width
                    st.subheader("Raw Data Table")
                    # Set the dataframe to use the full container width without any custom CSS
                    # This will make the table use the full width of its container
                    st.dataframe(df, height=600, use_container_width=True)
                    
                    # Display data info in tabular manner
                    st.subheader("Data Information")
                    
                    # Create a more structured table for data information
                    col_types = df.dtypes.value_counts()
                    missing_data = df.isnull().sum().sum()
                    
                    # Data info table
                    info_data = {
                        "Attribute": [
                            "Rows", 
                            "Columns", 
                            "Total Data Points",
                            "Missing Values",
                            "Numeric Columns",
                            "Text/Categorical Columns",
                            "Date/Time Columns",
                            "Boolean Columns"
                        ],
                        "Value": [
                            df.shape[0],
                            df.shape[1],
                            df.shape[0] * df.shape[1],
                            missing_data,
                            len(df.select_dtypes(include=['int64', 'float64']).columns),
                            len(df.select_dtypes(include=['object']).columns),
                            len(df.select_dtypes(include=['datetime64']).columns),
                            len(df.select_dtypes(include=['bool']).columns)
                        ]
                    }
                    
                    st.table(pd.DataFrame(info_data))
                    
                    if st.button("Proceed to Data Management"):
                        st.session_state.current_step = 2
                        # Add JavaScript to scroll to top after rerun
                        st.markdown(
                            """
                            <script>
                                window.scrollTo(0, 0);
                            </script>
                            """,
                            unsafe_allow_html=True
                        )
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        with sample_col:
            st.subheader("Use Sample Data")
            st.write("Don't have a dataset? Use our sample data to explore the application.")
            
            # Add a sample data option for demonstration
            if st.button("Use Sample Data (Iris Dataset)"):
                from sklearn.datasets import load_iris
                iris = load_iris()
                iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
                iris_df['target'] = iris.target
                iris_df['species'] = iris_df['target'].map({
                    0: 'setosa',
                    1: 'versicolor',
                    2: 'virginica'
                })
                # Reset index to start from 1 instead of 0
                iris_df.index = iris_df.index + 1
                
                st.session_state.data = iris_df
                st.session_state.filename = "iris_dataset.csv"
                st.success("Loaded sample Iris dataset")
                
                # Show the complete raw data table with fixed width
                st.subheader("Raw Data Table")
                # Set the dataframe to use the full container width without any custom CSS
                # This will make the table use the full width of its container
                st.dataframe(iris_df, height=600, use_container_width=True)
                
                # Display data info in tabular manner
                st.subheader("Data Information")
                
                # Create a more structured table for data information
                col_types = iris_df.dtypes.value_counts()
                missing_data = iris_df.isnull().sum().sum()
                
                # Data info table
                info_data = {
                    "Attribute": [
                        "Rows", 
                        "Columns", 
                        "Total Data Points",
                        "Missing Values",
                        "Numeric Columns",
                        "Text/Categorical Columns",
                        "Date/Time Columns",
                        "Boolean Columns"
                    ],
                    "Value": [
                        iris_df.shape[0],
                        iris_df.shape[1],
                        iris_df.shape[0] * iris_df.shape[1],
                        missing_data,
                        len(iris_df.select_dtypes(include=['int64', 'float64']).columns),
                        len(iris_df.select_dtypes(include=['object']).columns),
                        len(iris_df.select_dtypes(include=['datetime64']).columns),
                        len(iris_df.select_dtypes(include=['bool']).columns)
                    ]
                }
                
                st.table(pd.DataFrame(info_data))
                
                if st.button("Proceed to Data Management", key="proceed_sample"):
                    st.session_state.current_step = 2
                    # Add JavaScript to scroll to top after rerun
                    st.markdown(
                        """
                        <script>
                            window.scrollTo(0, 0);
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                    st.rerun()
    
    elif st.session_state.current_step == 2:
        # Step 2: Data Management
        st.header("2. Data Management")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Create tabs for different data management approaches
            tab1, tab2 = st.tabs(["Excel Drag & Drop Config", "Basic Column Selection"])
            
            # Tab 1: Excel Drag & Drop Config
            with tab1:
                # Use the drag & drop UI without introductory text
                display_drag_drop_ui()
                
                # Check if configuration is complete
                if 'configuration_complete' in st.session_state and st.session_state.configuration_complete:
                    # When configuration is complete, update selected columns with configured columns
                    
                    # Add age groups if age column is specified
                    if 'age_column' in st.session_state and st.session_state.age_column:
                        age_column = st.session_state.age_column['name']
                        age_type = st.session_state.age_type if 'age_type' in st.session_state else "years"
                        
                        # Add age column to selected columns
                        if age_column not in st.session_state.selected_columns:
                            st.session_state.selected_columns.append(age_column)
                        
                        # Age groups are now generated in drag_drop_ui.py when clicking Complete Configuration
                        # This section is intentionally disabled to prevent premature age group generation
                        pass
                    
                    # Add gender column
                    if 'gender_column' in st.session_state and st.session_state.gender_column:
                        if st.session_state.gender_column['name'] not in st.session_state.selected_columns:
                            st.session_state.selected_columns.append(st.session_state.gender_column['name'])
                    
                    # Add master columns
                    if 'master_columns' in st.session_state:
                        for col in st.session_state.master_columns:
                            if col['name'] not in st.session_state.selected_columns:
                                st.session_state.selected_columns.append(col['name'])
                    
                    # Add pair columns
                    if 'pair_areas' in st.session_state:
                        for pair in st.session_state.pair_areas:
                            for area_type in ['pre', 'post']:
                                for col in pair[area_type]:
                                    if col['name'] not in st.session_state.selected_columns:
                                        st.session_state.selected_columns.append(col['name'])
            
            # Tab 2: Basic Column Selection
            with tab2:
                # Display full dataframe with increased height to show all rows
                st.subheader("Full Dataset")
                st.dataframe(df, height=600)
                
                # Column selection for further analysis
                st.subheader("Select Columns for Analysis")
                all_cols = df.columns.tolist()
                st.session_state.selected_columns = st.multiselect(
                    "Select columns you want to include in your analysis:",
                    all_cols,
                    default=st.session_state.selected_columns if st.session_state.selected_columns else all_cols[:5]
                )
                
                if st.session_state.selected_columns:
                    # Show preview of selected columns
                    st.subheader("Preview of Selected Data")
                    st.dataframe(df[st.session_state.selected_columns].head())
                    
                    # Display column information
                    st.subheader("Column Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Numerical Columns:")
                        numerical_cols = [col for col in st.session_state.selected_columns if pd.api.types.is_numeric_dtype(df[col])]
                        if numerical_cols:
                            st.write(", ".join(numerical_cols))
                        else:
                            st.write("No numerical columns selected")
                    
                    with col2:
                        st.write("Categorical Columns:")
                        categorical_cols = [col for col in st.session_state.selected_columns if not pd.api.types.is_numeric_dtype(df[col])]
                        if categorical_cols:
                            st.write(", ".join(categorical_cols))
                        else:
                            st.write("No categorical columns selected")
                    
                    # Check for missing values
                    st.subheader("Missing Values")
                    missing_values = df[st.session_state.selected_columns].isnull().sum()
                    if missing_values.sum() > 0:
                        st.write("Columns with missing values:")
                        missing_df = pd.DataFrame({
                            'Column': missing_values.index,
                            'Missing Values': missing_values.values,
                            'Percentage': (missing_values.values / len(df) * 100).round(2)
                        })
                        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
                        st.table(missing_df)
                        
                        # Option to handle missing values
                        st.subheader("Handle Missing Values")
                        missing_strategy = st.selectbox(
                            "Select strategy for handling missing values:",
                            ["None (Keep as is)", "Drop rows with missing values", "Fill with mean/mode"]
                        )
                        
                        if missing_strategy != "None (Keep as is)" and st.button("Apply Missing Value Strategy"):
                            if missing_strategy == "Drop rows with missing values":
                                df_clean = df.dropna(subset=st.session_state.selected_columns)
                                st.session_state.data = df_clean
                                st.success(f"Dropped rows with missing values. Remaining rows: {len(df_clean)}")
                                st.dataframe(df_clean[st.session_state.selected_columns].head())
                            
                            elif missing_strategy == "Fill with mean/mode":
                                df_filled = df.copy()
                                for col in st.session_state.selected_columns:
                                    if pd.api.types.is_numeric_dtype(df[col]):
                                        # Fill numeric columns with mean
                                        fill_value = df[col].mean()
                                        df_filled[col] = df_filled[col].fillna(fill_value)
                                        st.info(f"Filled '{col}' with mean: {fill_value:.2f}")
                                    else:
                                        # Fill categorical columns with mode
                                        fill_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                                        df_filled[col] = df_filled[col].fillna(fill_value)
                                        st.info(f"Filled '{col}' with mode: {fill_value}")
                                
                                st.session_state.data = df_filled
                                st.success("Filled missing values with mean/mode.")
                                st.dataframe(df_filled[st.session_state.selected_columns].head())
                    else:
                        st.write("No missing values found in the selected columns.")
                    
                    # Data types
                    st.subheader("Data Types")
                    dtypes_df = pd.DataFrame({
                        'Column': [col for col in st.session_state.selected_columns],
                        'Data Type': [str(df[col].dtype) for col in st.session_state.selected_columns]  # Convert dtype to string to prevent Arrow conversion issues
                    })
                    st.table(dtypes_df)
                else:
                    st.warning("Please select at least one column for analysis.")
            
            # Final dataset section is now moved into the drag & drop UI
            # and appears only after clicking "Complete Configuration"
            
            # Determine if we have master, age, age group, gender, pairs columns to show
            display_columns = []
            
            # Add master columns if available
            if 'master_columns' in st.session_state and st.session_state.master_columns:
                master_cols = [col['name'] for col in st.session_state.master_columns]
                display_columns.extend(master_cols)
                
            # Add age column if available
            if 'age_column' in st.session_state and st.session_state.age_column:
                age_col = st.session_state.age_column['name']
                if age_col not in display_columns:
                    display_columns.append(age_col)
                
                # Add only one Age Group column if any exists in the dataframe
                # Prioritize the Generated Age Group (our dynamically created one)
                age_group_cols = [col for col in df.columns if 'age group' in col.lower()]
                if age_group_cols:
                    # If we have our generated column, use that
                    if 'Generated Age Group' in age_group_cols:
                        display_columns.append('Generated Age Group')
                    # Otherwise use the first available age group column
                    else:
                        display_columns.append(age_group_cols[0])
            
            # Add gender column if available
            if 'gender_column' in st.session_state and st.session_state.gender_column:
                gender_col = st.session_state.gender_column['name']
                if gender_col not in display_columns:
                    display_columns.append(gender_col)
            
            # Add pair columns if available
            if 'pair_areas' in st.session_state and st.session_state.pair_areas:
                for pair in st.session_state.pair_areas:
                    for area_type in ['pre', 'post']:
                        for col in pair[area_type]:
                            if col['name'] not in display_columns:
                                display_columns.append(col['name'])
            
            # This section has been moved to drag_drop_ui.py and appears
            # only after the Complete Configuration button is clicked
            
            # Make sure we're using all selected columns
            all_display_columns = st.session_state.selected_columns.copy()
            
            # Add any display columns that aren't already in the selected columns
            existing_display_columns = [col for col in display_columns if col in df.columns]
            for col in existing_display_columns:
                if col not in all_display_columns:
                    all_display_columns.append(col)
            
            if all_display_columns:
                # Create a styled dataframe with background colors for different column types
                styled_df = df[all_display_columns].head(10).style
                
                # Apply background colors based on column types
                def highlight_columns(x):
                    styles = pd.DataFrame('', index=x.index, columns=x.columns)
                    
                    # Master columns (light blue)
                    if 'master_columns' in st.session_state and st.session_state.master_columns:
                        master_cols = [col['name'] for col in st.session_state.master_columns]
                        for col in master_cols:
                            if col in x.columns:
                                styles[col] = 'background-color: #e6f2ff'
                    
                    # Age column (light yellow)
                    if 'age_column' in st.session_state and st.session_state.age_column:
                        age_col = st.session_state.age_column['name']
                        if age_col in x.columns:
                            styles[age_col] = 'background-color: #ffffcc'
                    
                    # Age group columns (light orange)
                    age_group_cols = [col for col in x.columns if 'age group' in col.lower() or 'generated age group' in col.lower()]
                    for col in age_group_cols:
                        styles[col] = 'background-color: #ffebcc'
                    
                    # Gender column (light pink)
                    if 'gender_column' in st.session_state and st.session_state.gender_column:
                        gender_col = st.session_state.gender_column['name']
                        if gender_col in x.columns:
                            styles[gender_col] = 'background-color: #ffdddd'
                    
                    # Pair columns (light green shades for pre, light purple for post)
                    if 'pair_areas' in st.session_state and st.session_state.pair_areas:
                        for idx, pair in enumerate(st.session_state.pair_areas):
                            # Pre columns (light green with varying shades)
                            green_intensity = 200 - (idx * 10) % 50  # Vary the shade for different pairs
                            for col_obj in pair['pre']:
                                col = col_obj['name']
                                if col in x.columns:
                                    styles[col] = f'background-color: rgba(144, {green_intensity}, 144, 0.3)'
                            
                            # Post columns (light purple with varying shades)
                            purple_intensity = 200 - (idx * 10) % 50  # Vary the shade for different pairs
                            for col_obj in pair['post']:
                                col = col_obj['name']
                                if col in x.columns:
                                    styles[col] = f'background-color: rgba({purple_intensity}, 144, 240, 0.3)'
                    
                    return styles
                
                styled_df = styled_df.apply(highlight_columns, axis=None)
                
                # Display the styled dataframe
                st.dataframe(styled_df, height=400)
                
                # Show legend for colors
                st.write("**Column Legend:**")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.markdown('<div style="background-color: #e6f2ff; padding: 5px;">Master Columns</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div style="background-color: #ffffcc; padding: 5px;">Age Column</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div style="background-color: #ffebcc; padding: 5px;">Age Group Column</div>', unsafe_allow_html=True)
                with col4:
                    st.markdown('<div style="background-color: #ffdddd; padding: 5px;">Gender Column</div>', unsafe_allow_html=True)
                with col5:
                    st.markdown('<div style="background-color: rgba(144, 200, 144, 0.3); padding: 5px;">Pre Pairs</div>', unsafe_allow_html=True)
                with col6:
                    st.markdown('<div style="background-color: rgba(200, 144, 240, 0.3); padding: 5px;">Post Pairs</div>', unsafe_allow_html=True)
            else:
                st.warning("No columns selected for analysis.")
            
            # Option to continue
            if st.button("Continue to Data Analysis"):
                if not st.session_state.selected_columns:
                    st.error("Please select at least one column for analysis before continuing.")
                else:
                    st.session_state.current_step = 3
                    # Add JavaScript to scroll to top after rerun
                    st.markdown(
                        """
                        <script>
                            window.scrollTo(0, 0);
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                    st.rerun()
    
    elif st.session_state.current_step == 3:
        # Step 3: Desired Data Analysis
        st.header("3. Desired Data Analysis")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # If columns weren't already selected, default to ALL columns
            if not st.session_state.selected_columns:
                st.session_state.selected_columns = df.columns.tolist()
            
            # Create analysis selection options
            st.subheader("Select Analysis Type")
            
            analysis_options = [
                "Descriptive Statistics",
                "Statistical Tests", 
                "Correlation Analysis",
                "Automated Analysis"
            ]
            
            st.session_state.analysis_type = st.radio(
                "What type of analysis would you like to perform?",
                analysis_options,
                index=analysis_options.index(st.session_state.analysis_type) if st.session_state.analysis_type in analysis_options else 0
            )
            
            if st.session_state.analysis_type == "Descriptive Statistics":
                st.write("You'll get summary statistics for your selected columns.")
            
            elif st.session_state.analysis_type == "Statistical Tests":
                st.write("You'll be able to perform hypothesis tests (t-tests, ANOVA) on your data.")
                
                # Allow user to select specific test
                test_options = ["t-Test", "ANOVA"]
                test_type = st.selectbox("Select specific test:", test_options)
                
                if test_type == "t-Test":
                    # t-test setup
                    # Use all columns for dropdown selection
                    all_cols = st.session_state.selected_columns
                    # Also keep track of numeric columns for validation
                    num_cols = [col for col in st.session_state.selected_columns if pd.api.types.is_numeric_dtype(df[col])]
                    ttest_settings = {}
                    
                    if all_cols:
                        ttest_settings['test_type'] = st.radio(
                            "Select t-test type:",
                            ["One-sample t-test", "Two-sample t-test"]
                        )
                    
                        if ttest_settings['test_type'] == "One-sample t-test":
                            ttest_settings['column'] = st.selectbox("Select column for test:", all_cols)
                            ttest_settings['mu'] = st.number_input("Population mean (μ₀):", value=0.0)
                            ttest_settings['alpha'] = st.slider("Significance level (α):", 0.01, 0.10, 0.05)
                        
                        else:  # Two-sample t-test
                            ttest_settings['method'] = st.radio(
                                "Select method:",
                                ["Compare two columns", "Compare groups within a column"]
                            )
                            
                            if ttest_settings['method'] == "Compare two columns":
                                ttest_settings['col1'] = st.selectbox("Select first column:", all_cols)
                                # Filter out the first column
                                remaining_cols = [col for col in all_cols if col != ttest_settings['col1']]
                                if remaining_cols:
                                    ttest_settings['col2'] = st.selectbox("Select second column:", remaining_cols)
                                    ttest_settings['equal_var'] = st.checkbox("Assume equal variances", value=False)
                                    ttest_settings['alpha'] = st.slider("Significance level (α):", 0.01, 0.10, 0.05, key="ttest_alpha2")
                                else:
                                    st.warning("Need at least two columns for two-sample t-test.")
                            
                            else:  # Compare groups
                                ttest_settings['num_col'] = st.selectbox("Select column to analyze:", all_cols)
                                # Use all columns for grouping, not just categorical ones
                                ttest_settings['group_col'] = st.selectbox("Select grouping column:", all_cols)
                                # Get unique values for the selected grouping column
                                unique_groups = df[ttest_settings['group_col']].dropna().unique().tolist()
                                
                                if len(unique_groups) >= 2:
                                    ttest_settings['group1'] = st.selectbox("Select first group:", unique_groups)
                                    # Filter out the first group
                                    remaining_groups = [g for g in unique_groups if g != ttest_settings['group1']]
                                    ttest_settings['group2'] = st.selectbox("Select second group:", remaining_groups)
                                    ttest_settings['equal_var'] = st.checkbox("Assume equal variances", value=False, key="ttest_eq_var2")
                                    ttest_settings['alpha'] = st.slider("Significance level (α):", 0.01, 0.10, 0.05, key="ttest_alpha3")
                                else:
                                    st.warning(f"Need at least two groups in {ttest_settings['group_col']} for two-sample t-test.")
                    else:
                        st.warning("No columns available for analysis.")
                    
                    # Store the settings in session state for later use
                    st.session_state.ttest_settings = ttest_settings
                
                elif test_type == "ANOVA":
                    # ANOVA setup
                    all_cols = st.session_state.selected_columns
                    anova_settings = {}
                    
                    if all_cols:
                        anova_settings['num_col'] = st.selectbox("Select column for analysis (dependent variable):", all_cols)
                        anova_settings['cat_col'] = st.selectbox("Select categorical column (factor):", all_cols)
                        
                        # Check if the categorical column has enough groups
                        unique_groups = df[anova_settings['cat_col']].dropna().unique()
                        if len(unique_groups) < 3:
                            st.warning(f"ANOVA requires at least 3 groups, but {anova_settings['cat_col']} only has {len(unique_groups)}.")
                        
                        anova_settings['alpha'] = st.slider("Significance level (α):", 0.01, 0.10, 0.05, key="anova_alpha")
                    else:
                        st.warning("No columns available for ANOVA.")
                    
                    # Store the settings in session state for later use
                    st.session_state.anova_settings = anova_settings
            
            elif st.session_state.analysis_type == "Correlation Analysis":
                st.write("You'll analyze relationships between numerical variables.")
                
                # Correlation settings
                all_cols = st.session_state.selected_columns
                corr_settings = {}
                
                if len(all_cols) >= 2:
                    corr_settings['columns'] = st.multiselect(
                        "Select columns for correlation analysis:",
                        all_cols,
                        default=all_cols  # Default to all columns 
                    )
                    
                    if len(corr_settings['columns']) >= 2:
                        corr_settings['method'] = st.selectbox(
                            "Correlation method:",
                            ["pearson", "spearman", "kendall"],
                            index=0
                        )
                        st.write("Pearson: Linear correlation, assumes normal distribution")
                        st.write("Spearman: Rank correlation, resistant to outliers")
                        st.write("Kendall: Another rank correlation, better for small samples")
                        
                        corr_settings['alpha'] = st.slider("Significance level (α):", 0.01, 0.10, 0.05, key="corr_alpha")
                    else:
                        st.warning("Please select at least two columns for correlation analysis.")
                else:
                    st.warning("Need at least two columns for correlation analysis.")
                
                # Store the settings in session state for later use
                st.session_state.corr_settings = corr_settings
                
            elif st.session_state.analysis_type == "Automated Analysis":
                st.write("We'll automatically analyze your data and suggest appropriate statistical tests.")
                
                # Show data structure analysis results
                data_analysis = analyze_data_structure(df)
                
                # Show data role detection (input/process/output)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Input Columns")
                    if 'input_columns' in data_analysis and data_analysis['input_columns']:
                        for col in data_analysis['input_columns'][:10]:
                            st.write(f"- {col}")
                        if len(data_analysis['input_columns']) > 10:
                            st.write(f"...and {len(data_analysis['input_columns']) - 10} more")
                    else:
                        st.write("No input columns detected")
                
                with col2:
                    st.subheader("Process Columns")
                    if 'process_columns' in data_analysis and data_analysis['process_columns']:
                        for col in data_analysis['process_columns'][:10]:
                            st.write(f"- {col}")
                        if len(data_analysis['process_columns']) > 10:
                            st.write(f"...and {len(data_analysis['process_columns']) - 10} more")
                    else:
                        st.write("No process columns detected")
                
                with col3:
                    st.subheader("Output Columns")
                    if 'output_columns' in data_analysis and data_analysis['output_columns']:
                        for col in data_analysis['output_columns'][:10]:
                            st.write(f"- {col}")
                        if len(data_analysis['output_columns']) > 10:
                            st.write(f"...and {len(data_analysis['output_columns']) - 10} more")
                    else:
                        st.write("No output columns detected")
                
                # Show detected calculation patterns
                if 'calculation_patterns' in data_analysis and data_analysis['calculation_patterns']:
                    st.subheader("Detected Calculation Patterns")
                    patterns = data_analysis['calculation_patterns']
                    
                    # Group patterns by type
                    pattern_types = {}
                    for pattern in patterns:
                        pattern_type = pattern['type']
                        if pattern_type not in pattern_types:
                            pattern_types[pattern_type] = []
                        pattern_types[pattern_type].append(pattern)
                    
                    # Show patterns by type
                    for pattern_type, type_patterns in pattern_types.items():
                        with st.expander(f"{pattern_type.replace('_', ' ').title()} ({len(type_patterns)} found)"):
                            for i, pattern in enumerate(type_patterns[:10]):
                                if pattern_type == 'pre_post_comparison':
                                    st.write(f"{i+1}. {pattern['pre_column']} → {pattern['post_column']}")
                                elif pattern_type == 'difference_calculation' or pattern_type == 'percentage_calculation':
                                    st.write(f"{i+1}. {' & '.join([str(col) for col in pattern['input_columns']])} → {pattern['output_column']}")
                                else:
                                    st.write(f"{i+1}. {pattern['output_column']}")
                            
                            if len(type_patterns) > 10:
                                st.write(f"...and {len(type_patterns) - 10} more")
                
                # Option to analyze all sheets if this is an Excel file
                if 'excel_file' in st.session_state:
                    st.subheader("Multi-Sheet Analysis")
                    analyze_all_sheets = st.checkbox("Analyze all sheets in the Excel file")
                    
                    if analyze_all_sheets:
                        try:
                            # Read the file with multiple sheets
                            uploaded_file = st.session_state.excel_file
                            from excel_analyzer import load_excel_file, analyze_excel_sheet
                            
                            with st.spinner("Loading and analyzing all sheets..."):
                                # Load all sheets from the Excel file
                                sheets_data, sheets_metadata = load_excel_file(uploaded_file)
                                
                                # Display sheet information
                                st.subheader("Excel File Structure")
                                st.write(f"Found {len(sheets_data)} sheets in the workbook.")
                                
                                # Analyze each sheet
                                all_sheets_analysis = {}
                                for sheet_name, sheet_df in sheets_data.items():
                                    with st.expander(f"Sheet: {sheet_name} ({sheet_df.shape[0]} rows, {sheet_df.shape[1]} columns)"):
                                        # Basic info about this sheet
                                        st.write(f"**Rows:** {sheet_df.shape[0]}, **Columns:** {sheet_df.shape[1]}")
                                        
                                        # Analyze data structure for this sheet
                                        sheet_analysis = analyze_data_structure(sheet_df)
                                        all_sheets_analysis[sheet_name] = sheet_analysis
                                        
                                        # Column summary for this sheet
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Numeric Columns:**")
                                            if sheet_analysis['numeric_columns']:
                                                for col in sheet_analysis['numeric_columns'][:10]:  # Limit display to first 10
                                                    st.write(f"- {col}")
                                                if len(sheet_analysis['numeric_columns']) > 10:
                                                    st.write(f"... and {len(sheet_analysis['numeric_columns']) - 10} more")
                                            else:
                                                st.write("None found")
                                        
                                        with col2:
                                            st.write("**Categorical Columns:**")
                                            if sheet_analysis['categorical_columns']:
                                                for col in sheet_analysis['categorical_columns'][:10]:  # Limit display to first 10
                                                    st.write(f"- {col}")
                                                if len(sheet_analysis['categorical_columns']) > 10:
                                                    st.write(f"... and {len(sheet_analysis['categorical_columns']) - 10} more")
                                            else:
                                                st.write("None found")
                                        
                                        # Display a preview of the data
                                        with st.expander("Data Preview"):
                                            st.dataframe(sheet_df.head(5))
                                        
                                        # Show suggested analyses for this sheet
                                        st.write("**Suggested Analyses:**")
                                        if 'suggested_analyses' in sheet_analysis:
                                            for analysis in sheet_analysis['suggested_analyses'][:5]:  # Limit to first 5 suggestions
                                                st.write(f"- {analysis.replace('_', ' ').title()}")
                                        else:
                                            st.write("No analyses suggested for this sheet.")
                                
                                # Cross-sheet analysis
                                st.subheader("Cross-Sheet Analysis")
                                
                                # Check for common columns across sheets
                                all_columns = {sheet: set(df.columns) for sheet, df in sheets_data.items()}
                                common_columns = set.intersection(*all_columns.values()) if all_columns else set()
                                
                                if common_columns:
                                    st.write(f"**Common Columns Across All Sheets ({len(common_columns)}):**")
                                    for col in sorted(common_columns)[:10]:  # Limit display to first 10
                                        st.write(f"- {col}")
                                    if len(common_columns) > 10:
                                        st.write(f"... and {len(common_columns) - 10} more")
                                    
                                    # Suggest cross-sheet comparison for a selected common column
                                    if len(common_columns) > 0:
                                        selected_column = st.selectbox("Select a common column for cross-sheet comparison:", sorted(common_columns))
                                        
                                        # Check if selected column is numeric
                                        numeric_in_all_sheets = True
                                        for sheet_name, sheet_df in sheets_data.items():
                                            if not pd.api.types.is_numeric_dtype(sheet_df[selected_column]):
                                                numeric_in_all_sheets = False
                                                break
                                        
                                        if numeric_in_all_sheets:
                                            # Create dataframe for comparison
                                            comparison_data = []
                                            for sheet_name, sheet_df in sheets_data.items():
                                                comparison_data.append({
                                                    'Sheet': sheet_name,
                                                    'Mean': sheet_df[selected_column].mean(),
                                                    'Median': sheet_df[selected_column].median(),
                                                    'Std Dev': sheet_df[selected_column].std(),
                                                    'Min': sheet_df[selected_column].min(),
                                                    'Max': sheet_df[selected_column].max(),
                                                    'Count': sheet_df[selected_column].count()
                                                })
                                            
                                            comparison_df = pd.DataFrame(comparison_data)
                                            st.write(f"**Comparison of {selected_column} Across Sheets:**")
                                            st.dataframe(comparison_df)
                                            
                                            # Create a visualization
                                            fig = px.bar(comparison_df, x='Sheet', y='Mean', 
                                                        error_y='Std Dev',
                                                        title=f"Mean of {selected_column} Across Sheets",
                                                        labels={'Mean': f'Mean {selected_column}'})
                                            st.plotly_chart(fig)
                                        else:
                                            st.write(f"Column '{selected_column}' is not numeric in all sheets.")
                                else:
                                    st.write("No common columns found across all sheets.")
                                
                                # Store the multi-sheet analysis results for later use
                                st.session_state.all_sheets_analysis = all_sheets_analysis
                                st.session_state.sheets_data = sheets_data
                        
                        except Exception as e:
                            st.error(f"Error analyzing all sheets: {str(e)}")
                        
                        # No need to continue with single-sheet analysis if we're analyzing all sheets
                        st.session_state.analysis_results = {
                            'type': 'multi_sheet',
                            'sheets_data': sheets_data,
                            'all_sheets_analysis': all_sheets_analysis
                        }
                        return
                
                # Continue with regular single-sheet analysis
                # Analyze data structure to get insights
                data_analysis = analyze_data_structure(df)
                
                # Show data characteristics
                st.subheader("Data Structure Analysis")
                
                # Basic data info
                st.write(f"**Rows:** {data_analysis['num_rows']}, **Columns:** {data_analysis['num_cols']}")
                
                # Column types summary
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Numeric Columns:**")
                    if data_analysis['numeric_columns']:
                        for col in data_analysis['numeric_columns']:
                            st.write(f"- {col}")
                    else:
                        st.write("None found")
                
                with col2:
                    st.write("**Categorical Columns:**")
                    if data_analysis['categorical_columns']:
                        for col in data_analysis['categorical_columns']:
                            st.write(f"- {col}")
                    else:
                        st.write("None found")
                
                # Possible grouping columns
                st.write("**Potential Grouping Variables:**")
                if data_analysis['possible_grouping_columns']:
                    for col in data_analysis['possible_grouping_columns']:
                        st.write(f"- {col}")
                else:
                    st.write("None found")
                
                # Pre/post data detection
                if data_analysis.get('has_pre_post_data', False):
                    st.write("**Pre/Post Data Detected!** Paired analysis is possible.")
                
                # Suggested analyses
                st.subheader("Suggested Analyses")
                
                # Store for automated analyses
                auto_analysis_options = []
                
                if 'suggested_analyses' in data_analysis:
                    for analysis in data_analysis['suggested_analyses']:
                        if analysis == 'correlation_analysis':
                            st.write("- **Correlation Analysis**: Examine relationships between numerical variables")
                            auto_analysis_options.append("Correlation Analysis")
                        
                        elif analysis == 'group_comparison':
                            st.write("- **Group Comparison**: Compare metrics across different groups")
                            auto_analysis_options.append("Group Comparison")
                        
                        elif analysis == 'anova':
                            st.write("- **ANOVA**: Test differences between group means")
                            auto_analysis_options.append("ANOVA")
                        
                        elif analysis == 'paired_t_test':
                            st.write("- **Paired t-test**: Compare before/after measurements")
                            auto_analysis_options.append("Paired t-test")
                        
                        elif analysis == 'regression_analysis':
                            st.write("- **Regression Analysis**: Model relationships between variables")
                            auto_analysis_options.append("Regression Analysis")
                        
                        elif analysis == 'clustering':
                            st.write("- **Clustering**: Find natural groupings in your data")
                            auto_analysis_options.append("Clustering")
                
                # Let user select analyses to run automatically
                if auto_analysis_options:
                    st.session_state.auto_analyses = st.multiselect(
                        "Select analyses to run automatically:",
                        auto_analysis_options,
                        default=auto_analysis_options[:3]  # Default to first 3 suggestions
                    )
                    
                    # Show options based on selected analyses
                    if "Group Comparison" in st.session_state.auto_analyses:
                        # Get potential grouping columns
                        group_cols = data_analysis.get('possible_grouping_columns', [])
                        if group_cols:
                            st.session_state.group_col = st.selectbox(
                                "Select grouping column for comparison:",
                                group_cols
                            )
                    
                    if "Paired t-test" in st.session_state.auto_analyses:
                        # Try to identify pre/post columns
                        pre_cols = [col for col in df.columns if 'pre' in col.lower()]
                        post_cols = [col for col in df.columns if 'post' in col.lower()]
                        
                        if pre_cols and post_cols:
                            st.session_state.pre_col = st.selectbox("Select Pre measurement column:", pre_cols)
                            
                            # Get potential matching post columns based on selected pre column
                            # Look for similar names (pre -> post)
                            matching_post = st.session_state.pre_col.lower().replace('pre', 'post')
                            matching_posts = [col for col in post_cols if col.lower() == matching_post]
                            
                            if matching_posts:
                                default_post = matching_posts[0]
                            else:
                                default_post = post_cols[0] if post_cols else None
                            
                            if default_post:
                                st.session_state.post_col = st.selectbox(
                                    "Select Post measurement column:", 
                                    post_cols,
                                    index=post_cols.index(default_post) if default_post in post_cols else 0
                                )
                    
                    if "Regression Analysis" in st.session_state.auto_analyses:
                        num_cols = data_analysis.get('numeric_columns', [])
                        if len(num_cols) >= 2:
                            st.session_state.target_col = st.selectbox(
                                "Select target variable (dependent variable):",
                                num_cols
                            )
                            
                            # Filter out the target column for predictors
                            predictor_options = [col for col in num_cols if col != st.session_state.target_col]
                            st.session_state.predictor_cols = st.multiselect(
                                "Select predictor variables:",
                                predictor_options,
                                default=predictor_options[:min(3, len(predictor_options))]  # Default to first 3
                            )
                else:
                    st.warning("No automatic analyses could be suggested based on your data structure.")
            
            # Button to proceed to data tables
            if st.button("Continue to Data Tables"):
                st.session_state.current_step = 4
                # Add JavaScript to scroll to top after rerun
                st.markdown(
                    """
                    <script>
                        window.scrollTo(0, 0);
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                st.rerun()
    
    elif st.session_state.current_step == 4:
        # Step 4: Tables of Data
        st.header("4. Tables of Data")
        
        if st.session_state.data is not None and st.session_state.selected_columns:
            df = st.session_state.data
            
            # Show results based on selected analysis type
            if st.session_state.analysis_type == "Descriptive Statistics":
                st.subheader("Descriptive Statistics")
                
                try:
                    # Generate and display descriptive statistics
                    # Get stats for all columns regardless of what was selected for display
                    stats_df = get_descriptive_stats(df)
                    
                    # Convert 'Mean ± Std Dev' to a string column to avoid Arrow conversion errors
                    for col in stats_df.columns:
                        if isinstance(stats_df.loc['Mean ± Std Dev', col], str):
                            # If Mean ± Std Dev is already a string, we need to convert the whole column to strings
                            # for proper display in the dataframe
                            stats_df[col] = stats_df[col].astype(str)
                    
                    # Display the raw data table first
                    st.subheader("Raw Data Table")
                    st.dataframe(df, height=600)
                    
                    st.subheader("Overall Descriptive Statistics")
                    st.dataframe(stats_df)
                    
                    # Master parameter analysis is performed further below
                    # The section here was intentionally removed to prevent duplicate calculations
                    
                    # Store results for later
                    st.session_state.analysis_results = {
                        'type': 'descriptive',
                        'stats_df': stats_df
                    }
                    
                    # Display additional statistics for all columns
                    # Get all selected columns
                    all_selected_cols = st.session_state.selected_columns
                    
                    # Check if we have master columns from configuration
                    master_columns = []
                    if 'configuration_complete' in st.session_state and st.session_state.configuration_complete:
                        if 'master_columns' in st.session_state:
                            # Get column names from the column objects
                            master_columns = [col['name'] for col in st.session_state.master_columns]
                            
                    # If we have master columns, perform master-specific analysis
                    if master_columns:
                        st.subheader("Master Parameter Analysis")
                        
                        # Analyze data grouped by master parameters
                        master_results = analyze_master_parameters(df, master_columns)
                        
                        if master_results:
                            # Display analysis for each master column
                            for master_col, results in master_results.items():
                                if results.get('type') == 'error':
                                    st.error(f"Error analyzing {master_col}: {results.get('error')}")
                                    continue
                                    
                                st.write(f"### Analysis by {master_col}")
                                col_type = results.get('type', 'unknown')
                                st.write(f"Column type: {col_type}")
                                
                                # Check if there's only one group
                                groups = results.get('groups', {})
                                if len(groups) == 1:
                                    group_name = list(groups.keys())[0]
                                    st.info(f"⚠️ Note: The dataset only contains one value for {master_col} ({group_name}). " +
                                           f"For comparative analysis, upload data with multiple {master_col} values.")
                                
                                # Display results for each group
                                for group_name, group_data in groups.items():
                                    with st.expander(f"{master_col} = {group_name} (n={group_data['count']})"):
                                        group_stats = group_data.get('stats')
                                        if group_stats is not None:
                                            # Convert to string for display
                                            for col in group_stats.columns:
                                                # Use the Unicode character for Mean ± Std Dev
                                                if 'Mean \u00B1 Std Dev' in group_stats.index and isinstance(group_stats.loc['Mean \u00B1 Std Dev', col], str):
                                                    group_stats[col] = group_stats[col].astype(str)
                                            
                                            st.dataframe(group_stats)
                                        else:
                                            st.write("No statistics available for this group.")
                        else:
                            st.info("No master parameter results were found. Make sure you have assigned master columns in the configuration.")
                    
                    # Calculate statistics for all columns
                    st.subheader("Additional Statistics for All Columns")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Skewness:")
                        # Create DataFrame with all columns
                        skew_data = []
                        for col in all_selected_cols:
                            try:
                                # Try to calculate skewness - will work for numerical and some other types
                                skew_value = df[col].skew()
                                skew_data.append({'Column': col, 'Skewness': skew_value})
                            except:
                                # For columns where skewness can't be calculated
                                skew_data.append({'Column': col, 'Skewness': float('nan')})
                        
                        skew_df = pd.DataFrame(skew_data)
                        # Convert all values to strings to prevent Arrow conversion errors
                        skew_df['Skewness'] = skew_df['Skewness'].apply(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")
                        st.dataframe(skew_df)
                    
                    with col2:
                        st.write("Kurtosis:")
                        # Create DataFrame with all columns
                        kurt_data = []
                        for col in all_selected_cols:
                            try:
                                # Try to calculate kurtosis - will work for numerical and some other types
                                kurt_value = df[col].kurtosis()
                                kurt_data.append({'Column': col, 'Kurtosis': kurt_value})
                            except:
                                # For columns where kurtosis can't be calculated
                                kurt_data.append({'Column': col, 'Kurtosis': float('nan')})
                        
                        kurt_df = pd.DataFrame(kurt_data)
                        # Convert all values to strings to prevent Arrow conversion errors
                        kurt_df['Kurtosis'] = kurt_df['Kurtosis'].apply(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")
                        st.dataframe(kurt_df)
                    
                    # We've already handled master parameter analysis above
                
                except Exception as e:
                    st.error(f"Error calculating statistics: {str(e)}")
            
            elif st.session_state.analysis_type == "Statistical Tests":
                if 'ttest_settings' in st.session_state and st.session_state.ttest_settings:
                    st.subheader("t-Test Results")
                    settings = st.session_state.ttest_settings
                    
                    try:
                        if settings.get('test_type') == "One-sample t-test":
                            result = perform_ttest(
                                df, 
                                settings['column'], 
                                test_type="one-sample", 
                                mu=settings['mu'], 
                                alpha=settings['alpha']
                            )
                            
                            # Display results
                            st.write(f"**Testing if the mean of {settings['column']} differs from {settings['mu']}**")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"Sample mean: {result['sample_mean']:.4f}")
                                st.write(f"t-statistic: {result['t_stat']:.4f}")
                                st.write(f"p-value: {result['p_value']:.4f}")
                            
                            with col2:
                                st.write(f"Degrees of freedom: {result['df']}")
                                st.write(f"95% Confidence Interval: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")
                            
                            # Interpretation
                            st.subheader("Interpretation")
                            if result['p_value'] < settings['alpha']:
                                st.write(f"With a p-value of {result['p_value']:.4f}, which is less than the significance level of {settings['alpha']}, we **reject** the null hypothesis.")
                                st.write(f"There is significant evidence that the true mean of {settings['column']} differs from {settings['mu']}.")
                            else:
                                st.write(f"With a p-value of {result['p_value']:.4f}, which is greater than the significance level of {settings['alpha']}, we **fail to reject** the null hypothesis.")
                                st.write(f"There is insufficient evidence that the true mean of {settings['column']} differs from {settings['mu']}.")
                            
                            # Store results for later
                            st.session_state.analysis_results = {
                                'type': 'ttest',
                                'test_type': 'one-sample',
                                'result': result
                            }
                        
                        elif settings.get('test_type') == "Two-sample t-test":
                            if settings.get('method') == "Compare two columns":
                                result = perform_ttest(
                                    df, 
                                    settings['col1'], 
                                    test_type="two-sample-cols", 
                                    col2=settings['col2'], 
                                    equal_var=settings['equal_var'], 
                                    alpha=settings['alpha']
                                )
                                
                                # Display results
                                st.write(f"**Comparing means of {settings['col1']} and {settings['col2']}**")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"Mean of {settings['col1']}: {result['mean1']:.4f}")
                                    st.write(f"Mean of {settings['col2']}: {result['mean2']:.4f}")
                                    st.write(f"Mean difference: {result['mean_diff']:.4f}")
                                
                                with col2:
                                    st.write(f"t-statistic: {result['t_stat']:.4f}")
                                    st.write(f"p-value: {result['p_value']:.4f}")
                                    st.write(f"Degrees of freedom: {result['df']:.1f}")
                                
                                st.write(f"95% Confidence Interval for difference: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")
                                
                                # Interpretation
                                st.subheader("Interpretation")
                                test_name = "Welch's t-test" if not settings['equal_var'] else "Student's t-test"
                                st.write(f"Using {test_name} ({'unequal' if not settings['equal_var'] else 'equal'} variances):")
                                
                                if result['p_value'] < settings['alpha']:
                                    st.write(f"With a p-value of {result['p_value']:.4f}, which is less than the significance level of {settings['alpha']}, we **reject** the null hypothesis.")
                                    st.write(f"There is significant evidence that the means of {settings['col1']} and {settings['col2']} differ.")
                                else:
                                    st.write(f"With a p-value of {result['p_value']:.4f}, which is greater than the significance level of {settings['alpha']}, we **fail to reject** the null hypothesis.")
                                    st.write(f"There is insufficient evidence that the means of {settings['col1']} and {settings['col2']} differ.")
                                
                                # Store results for later
                                st.session_state.analysis_results = {
                                    'type': 'ttest',
                                    'test_type': 'two-sample-cols',
                                    'result': result
                                }
                            
                            elif settings.get('method') == "Compare groups within a column":
                                result = perform_ttest(
                                    df, 
                                    settings['num_col'], 
                                    test_type="two-sample-groups", 
                                    group_col=settings['group_col'],
                                    group1=settings['group1'],
                                    group2=settings['group2'],
                                    equal_var=settings['equal_var'], 
                                    alpha=settings['alpha']
                                )
                                
                                # Display results
                                st.write(f"**Comparing {settings['num_col']} between {settings['group1']} and {settings['group2']} groups**")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"Mean for {settings['group1']}: {result['mean1']:.4f}")
                                    st.write(f"Mean for {settings['group2']}: {result['mean2']:.4f}")
                                    st.write(f"Mean difference: {result['mean_diff']:.4f}")
                                
                                with col2:
                                    st.write(f"t-statistic: {result['t_stat']:.4f}")
                                    st.write(f"p-value: {result['p_value']:.4f}")
                                    st.write(f"Degrees of freedom: {result['df']:.1f}")
                                
                                st.write(f"95% Confidence Interval for difference: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")
                                
                                # Interpretation
                                st.subheader("Interpretation")
                                test_name = "Welch's t-test" if not settings['equal_var'] else "Student's t-test"
                                st.write(f"Using {test_name} ({'unequal' if not settings['equal_var'] else 'equal'} variances):")
                                
                                if result['p_value'] < settings['alpha']:
                                    st.write(f"With a p-value of {result['p_value']:.4f}, which is less than the significance level of {settings['alpha']}, we **reject** the null hypothesis.")
                                    st.write(f"There is significant evidence that the mean {settings['num_col']} differs between {settings['group1']} and {settings['group2']} groups.")
                                else:
                                    st.write(f"With a p-value of {result['p_value']:.4f}, which is greater than the significance level of {settings['alpha']}, we **fail to reject** the null hypothesis.")
                                    st.write(f"There is insufficient evidence that the mean {settings['num_col']} differs between {settings['group1']} and {settings['group2']} groups.")
                                
                                # Store results for later
                                st.session_state.analysis_results = {
                                    'type': 'ttest',
                                    'test_type': 'two-sample-groups',
                                    'result': result
                                }
                    
                    except Exception as e:
                        st.error(f"Error performing t-test: {str(e)}")
                
                elif 'anova_settings' in st.session_state and st.session_state.anova_settings:
                    st.subheader("ANOVA Results")
                    settings = st.session_state.anova_settings
                    
                    try:
                        if 'num_col' in settings and 'cat_col' in settings:
                            result = perform_anova(
                                df, 
                                settings['num_col'], 
                                settings['cat_col'], 
                                alpha=settings['alpha']
                            )
                            
                            # Display ANOVA table
                            st.write(f"**One-way ANOVA: Effect of {settings['cat_col']} on {settings['num_col']}**")
                            st.dataframe(result['anova_table'])
                            
                            # Display group statistics
                            st.subheader("Group Statistics")
                            st.dataframe(result['group_stats'])
                            
                            # Interpretation
                            st.subheader("Interpretation")
                            if result['p_value'] < settings['alpha']:
                                st.write(f"With a p-value of {result['p_value']:.4f}, which is less than the significance level of {settings['alpha']}, we **reject** the null hypothesis.")
                                st.write(f"There is significant evidence that the mean of {settings['num_col']} differs across at least one of the {settings['cat_col']} groups.")
                            else:
                                st.write(f"With a p-value of {result['p_value']:.4f}, which is greater than the significance level of {settings['alpha']}, we **fail to reject** the null hypothesis.")
                                st.write(f"There is insufficient evidence that the mean of {settings['num_col']} differs across the {settings['cat_col']} groups.")
                            
                            # Store results for later
                            st.session_state.analysis_results = {
                                'type': 'anova',
                                'result': result
                            }
                    
                    except Exception as e:
                        st.error(f"Error performing ANOVA: {str(e)}")
            
            elif st.session_state.analysis_type == "Correlation Analysis":
                if 'corr_settings' in st.session_state and st.session_state.corr_settings:
                    st.subheader("Correlation Analysis Results")
                    settings = st.session_state.corr_settings
                    
                    try:
                        if 'columns' in settings and len(settings['columns']) >= 2:
                            result = perform_correlation_analysis(
                                df, 
                                settings['columns'], 
                                method=settings['method'], 
                                alpha=settings['alpha']
                            )
                            
                            # Display correlation matrix
                            st.subheader(f"{settings['method'].capitalize()} Correlation Matrix")
                            st.dataframe(result['corr_matrix'])
                            
                            # Display significant correlations
                            st.subheader("Significant Correlations")
                            if not result['significant_corrs'].empty:
                                st.dataframe(result['significant_corrs'])
                            else:
                                st.write("No significant correlations found at the specified significance level.")
                            
                            # Store results for later
                            st.session_state.analysis_results = {
                                'type': 'correlation',
                                'result': result
                            }
                    
                    except Exception as e:
                        st.error(f"Error performing correlation analysis: {str(e)}")
            
            # Button to proceed to visualizations
            if st.button("Continue to Visualizations"):
                st.session_state.current_step = 5
                # Add JavaScript to scroll to top after rerun
                st.markdown(
                    """
                    <script>
                        window.scrollTo(0, 0);
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                st.rerun()
    
    elif st.session_state.current_step == 5:
        # Step 5: Graph/Chart Preparation
        st.markdown('<div id="visualizations"></div>', unsafe_allow_html=True)
        st.header("5. Graph/Chart Preparation")
        
        if st.session_state.data is not None and st.session_state.selected_columns:
            df = st.session_state.data
            
            # Visualization type selector
            st.subheader("Select Visualization Type")
            
            # Check if we have master columns from configuration
            master_columns = []
            if 'configuration_complete' in st.session_state and st.session_state.configuration_complete:
                if 'master_columns' in st.session_state:
                    # Get column names from the column objects
                    master_columns = [col['name'] for col in st.session_state.master_columns]
            
            viz_options = [
                "Histogram", 
                "Box Plot", 
                "Scatter Plot", 
                "Correlation Heatmap", 
                "Distribution Plot",
                "Bar Chart",
                "Line Chart"
            ]
            
            # Add Master Analysis as an option if we have master columns
            if master_columns:
                viz_options.insert(0, "Master Parameter Analysis")
            
            st.session_state.visualization_type = st.selectbox(
                "Choose the type of visualization you want to create:",
                viz_options,
                index=viz_options.index(st.session_state.visualization_type) if st.session_state.visualization_type in viz_options else 0
            )
            
            # Create appropriate visualization based on user selection
            if st.session_state.visualization_type == "Master Parameter Analysis":
                st.subheader("Master Parameter Analysis Visualization")
                
                # Check if we have master columns
                if not master_columns:
                    st.warning("No master columns available. Please go back to the configuration and assign master columns.")
                else:
                    # Let user select which master column to analyze
                    selected_master = st.selectbox("Select master parameter for analysis:", master_columns)
                    
                    # Analyze data grouped by the selected master parameter
                    master_results = analyze_master_parameters(df, [selected_master])
                    
                    if not master_results or selected_master not in master_results:
                        st.error(f"Error analyzing master parameter {selected_master}.")
                    else:
                        results = master_results[selected_master]
                        col_type = results.get('type', 'unknown')
                        st.write(f"Column type: {col_type}")
                        
                        # Check if there's only one group
                        groups = results.get('groups', {})
                        if len(groups) == 1:
                            group_name = list(groups.keys())[0]
                            st.info(f"⚠️ Note: The dataset only contains one value for {selected_master} ({group_name}). " +
                                   f"For comparative analysis, upload data with multiple {selected_master} values.")
                        
                        # Chart type selection
                        chart_type = st.radio(
                            "Select chart type for master parameter analysis:",
                            ["Group Statistics Comparison", "Distribution by Groups", "Pre-Post Analysis by Group"]
                        )
                        
                        # Select a numeric column to visualize across groups
                        num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != selected_master]
                        if not num_cols:
                            st.warning("No numerical columns available for visualization.")
                        else:
                            if chart_type in ["Group Statistics Comparison", "Distribution by Groups"]:
                                selected_numeric = st.selectbox("Select numerical column to visualize across groups:", num_cols)
                            elif chart_type == "Pre-Post Analysis by Group":
                                # For pre/post analysis
                                pre_post_pairs = st.session_state.get('pre_post_pairs', [])
                                
                                # Let user select pre/post columns
                                col1, col2 = st.columns(2)
                                with col1:
                                    pre_col = st.selectbox("Select Pre column:", num_cols, key="pre_col_selector")
                                with col2:
                                    post_col = st.selectbox("Select Post column:", num_cols, key="post_col_selector")
                            
                            # Handle different chart types
                            if chart_type in ["Group Statistics Comparison", "Distribution by Groups"]:
                                # Create summary dataframe for visualization
                                summary_data = []
                                for group_name, group_data in groups.items():
                                    group_stats = group_data.get('stats')
                                    if group_stats is not None and selected_numeric in group_stats.columns:
                                        # Extract statistics - try to get actual numeric values
                                        try:
                                            mean_val = float(group_stats.loc['Mean', selected_numeric])
                                            std_val = float(group_stats.loc['Std Dev', selected_numeric]) if 'Std Dev' in group_stats.index else 0
                                            median_val = float(group_stats.loc['Median', selected_numeric]) if 'Median' in group_stats.index else mean_val
                                            min_val = float(group_stats.loc['Min', selected_numeric]) if 'Min' in group_stats.index else 0
                                            max_val = float(group_stats.loc['Max', selected_numeric]) if 'Max' in group_stats.index else 0
                                            
                                            if pd.notna(mean_val):
                                                summary_data.append({
                                                    'Group': str(group_name),
                                                    'Mean': mean_val,
                                                    'Std Dev': std_val,
                                                    'Median': median_val,
                                                    'Min': min_val,
                                                    'Max': max_val,
                                                    'Count': group_data['count']
                                                })
                                        except (ValueError, TypeError):
                                            # Skip if we can't extract a numeric value
                                            pass
                            
                            elif chart_type == "Pre-Post Analysis by Group":
                                # Handle pre-post analysis by group
                                if 'pre_col' not in locals() or 'post_col' not in locals():
                                    st.warning("Please select Pre and Post columns for analysis.")
                                else:
                                    # Create a dataframe for pre-post comparison
                                    pre_post_data = []
                                    
                                    # Calculate pre-post stats for each group
                                    for group_name, group_data in groups.items():
                                        group_df = group_data.get('data')
                                        if group_df is not None and pre_col in group_df.columns and post_col in group_df.columns:
                                            pre_mean = group_df[pre_col].mean()
                                            post_mean = group_df[post_col].mean()
                                            pre_std = group_df[pre_col].std()
                                            post_std = group_df[post_col].std()
                                            count = len(group_df)
                                            # Calculate percent change
                                            if pre_mean != 0:
                                                pct_change = ((post_mean - pre_mean) / pre_mean) * 100
                                            else:
                                                pct_change = 0
                                                
                                            # Calculate p-value for pre vs post
                                            try:
                                                from scipy import stats
                                                t_stat, p_value = stats.ttest_rel(group_df[pre_col], group_df[post_col])
                                            except:
                                                p_value = float('nan')
                                            
                                            pre_post_data.append({
                                                'Group': str(group_name),
                                                'Pre Mean': pre_mean,
                                                'Pre StdDev': pre_std,
                                                'Post Mean': post_mean,
                                                'Post StdDev': post_std,
                                                'Change %': pct_change,
                                                'P-value': p_value,
                                                'Count': count
                                            })
                                    
                                    summary_data = pre_post_data
                            
                            if not summary_data:
                                st.warning(f"Could not extract valid statistics for {selected_numeric} across {selected_master} groups.")
                            else:
                                summary_df = pd.DataFrame(summary_data)
                                
                                # Visualization type
                                viz_subtype = st.radio(
                                    "Select visualization type:",
                                    ["Bar Chart", "Box Plot"]
                                )
                                
                                if viz_subtype == "Bar Chart":
                                    import plotly.express as px
                                    import plotly.graph_objects as go
                                    
                                    # Let user select stat to display based on chart type
                                    if chart_type in ["Group Statistics Comparison", "Distribution by Groups"]:
                                        stat_options = ["Mean", "Mean with StdDev", "Median", "Min/Max"]
                                        stat_display = st.selectbox("Choose statistic to display:", stat_options)
                                    elif chart_type == "Pre-Post Analysis by Group":
                                        stat_options = ["Pre-Post Means", "Percent Reduction", "Pre vs Post with P-values"]
                                        stat_display = st.selectbox("Choose pre-post analysis display:", stat_options)
                                    
                                    if stat_display == "Mean":
                                        # Create bar chart comparing means across groups
                                        fig = px.bar(
                                            summary_df,
                                            x='Group',
                                            y='Mean',
                                            title=f"Mean {selected_numeric} by {selected_master}",
                                            labels={'Group': selected_master, 'Mean': f'Mean {selected_numeric}'},
                                            text='Count',  # Show count on bars
                                            color='Group',
                                            color_discrete_sequence=px.colors.qualitative.Plotly
                                        )
                                        
                                        # Add count labels above bars
                                        fig.update_traces(texttemplate='n=%{text}', textposition='outside')
                                    
                                    elif stat_display == "Mean with StdDev":
                                        # Create bar chart with error bars
                                        fig = px.bar(
                                            summary_df,
                                            x='Group',
                                            y='Mean',
                                            error_y='Std Dev',
                                            title=f"Mean ± StdDev of {selected_numeric} by {selected_master}",
                                            labels={'Group': selected_master, 'Mean': f'Mean {selected_numeric}'},
                                            color='Group',
                                            color_discrete_sequence=px.colors.qualitative.Plotly
                                        )
                                        
                                        # Add formatted Mean ± StdDev as text
                                        for i, row in summary_df.iterrows():
                                            fig.add_annotation(
                                                x=row['Group'],
                                                y=row['Mean'] + row['Std Dev'] + 0.2,  # Position above error bar
                                                text=f"{row['Mean']:.2f} ± {row['Std Dev']:.2f} (n={row['Count']})",
                                                showarrow=False
                                            )
                                    
                                    elif stat_display == "Median":
                                        # Create bar chart comparing medians across groups
                                        fig = px.bar(
                                            summary_df,
                                            x='Group',
                                            y='Median',
                                            title=f"Median {selected_numeric} by {selected_master}",
                                            labels={'Group': selected_master, 'Median': f'Median {selected_numeric}'},
                                            text='Count',
                                            color='Group',
                                            color_discrete_sequence=px.colors.qualitative.Plotly
                                        )
                                        
                                        fig.update_traces(texttemplate='n=%{text}', textposition='outside')
                                    
                                    elif stat_display == "Min/Max":
                                        # Create a figure with min/max range
                                        fig = go.Figure()
                                        
                                        # Add a bar for each group showing min/max range
                                        for i, row in summary_df.iterrows():
                                            fig.add_trace(go.Bar(
                                                x=[row['Group']],
                                                y=[row['Max'] - row['Min']],
                                                base=[row['Min']],
                                                name=row['Group'],
                                                text=[f"Min: {row['Min']:.2f}<br>Max: {row['Max']:.2f}<br>n={row['Count']}"],
                                                hoverinfo='text'
                                            ))
                                        
                                        fig.update_layout(
                                            title=f"Min/Max Range of {selected_numeric} by {selected_master}",
                                            xaxis_title=selected_master,
                                            yaxis_title=f"{selected_numeric} Range"
                                        )
                                    
                                    # Pre-Post analysis display options
                                    elif stat_display == "Pre-Post Means":
                                        # Create a grouped bar chart for pre and post means
                                        fig = go.Figure()
                                        
                                        # Get pre and post column names
                                        pre_name = pre_col
                                        post_name = post_col
                                        
                                        # Create the grouped bar chart - pre values
                                        fig.add_trace(go.Bar(
                                            x=summary_df['Group'],
                                            y=[row['Pre Mean'] for i, row in summary_df.iterrows()],
                                            name=f"{pre_name}",
                                            text=[f"{row['Pre Mean']:.2f}<br>n={row['Count']}" for i, row in summary_df.iterrows()],
                                            textposition='auto',
                                            marker_color='#2ca02c'  # Green
                                        ))
                                        
                                        # Add post values
                                        fig.add_trace(go.Bar(
                                            x=summary_df['Group'],
                                            y=[row['Post Mean'] for i, row in summary_df.iterrows()],
                                            name=f"{post_name}",
                                            text=[f"{row['Post Mean']:.2f}<br>n={row['Count']}" for i, row in summary_df.iterrows()],
                                            textposition='auto',
                                            marker_color='#d62728'  # Red
                                        ))
                                        
                                        # Update layout
                                        fig.update_layout(
                                            title=f"Average Pre and Post {pre_name.replace('Pre', '')} by {selected_master}",
                                            xaxis_title=selected_master,
                                            yaxis_title=f"Value",
                                            barmode='group'
                                        )
                                        
                                    elif stat_display == "Percent Reduction":
                                        # Create a bar chart showing percent reduction
                                        fig = go.Figure()
                                        
                                        # Add percent change bars
                                        fig.add_trace(go.Bar(
                                            x=summary_df['Group'],
                                            y=[-row['Change %'] for i, row in summary_df.iterrows()],  # Negate to show reduction as positive
                                            text=[f"{abs(row['Change %']):.2f}%" for i, row in summary_df.iterrows()],
                                            textposition='outside',
                                            marker_color='#1f77b4'  # Blue
                                        ))
                                        
                                        # Update layout
                                        fig.update_layout(
                                            title=f"Percentage Reduction in {pre_col.replace('Pre', '')} by {selected_master}",
                                            xaxis_title=selected_master,
                                            yaxis_title="Reduction (%)"
                                        )
                                        
                                    elif stat_display == "Pre vs Post with P-values":
                                        # Create a grouped bar chart for pre and post with p-values annotation
                                        fig = go.Figure()
                                        
                                        # Get pre and post column names
                                        pre_name = pre_col
                                        post_name = post_col
                                        
                                        # Create the grouped bar chart - pre values
                                        fig.add_trace(go.Bar(
                                            x=summary_df['Group'],
                                            y=[row['Pre Mean'] for i, row in summary_df.iterrows()],
                                            name=f"{pre_name}",
                                            error_y=dict(
                                                type='data',
                                                array=[row['Pre StdDev'] for i, row in summary_df.iterrows()],
                                                visible=True
                                            ),
                                            marker_color='#2ca02c'  # Green
                                        ))
                                        
                                        # Add post values
                                        fig.add_trace(go.Bar(
                                            x=summary_df['Group'],
                                            y=[row['Post Mean'] for i, row in summary_df.iterrows()],
                                            name=f"{post_name}",
                                            error_y=dict(
                                                type='data',
                                                array=[row['Post StdDev'] for i, row in summary_df.iterrows()],
                                                visible=True
                                            ),
                                            marker_color='#d62728'  # Red
                                        ))
                                        
                                        # Add p-value annotations
                                        for i, row in summary_df.iterrows():
                                            p_val = row['P-value']
                                            sig_text = ""
                                            if not pd.isna(p_val):
                                                if p_val < 0.001:
                                                    sig_text = "p<0.001***"
                                                elif p_val < 0.01:
                                                    sig_text = f"p={p_val:.3f}**"
                                                elif p_val < 0.05:
                                                    sig_text = f"p={p_val:.3f}*"
                                                else:
                                                    sig_text = f"p={p_val:.3f}"
                                            
                                            fig.add_annotation(
                                                x=row['Group'],
                                                y=max(row['Pre Mean'], row['Post Mean']) + 
                                                   max(row['Pre StdDev'], row['Post StdDev']) + 0.5,
                                                text=sig_text,
                                                showarrow=False,
                                                font=dict(size=10)
                                            )
                                        
                                        # Update layout
                                        fig.update_layout(
                                            title=f"Pre vs Post {pre_name.replace('Pre', '')} with P-values by {selected_master}",
                                            xaxis_title=selected_master,
                                            yaxis_title=f"Value",
                                            barmode='group'
                                        )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Store visualization for output
                                    st.session_state.visualization = {
                                        'type': 'master_bar_chart',
                                        'figure': fig,
                                        'settings': {
                                            'master_column': selected_master,
                                            'numeric_column': selected_numeric
                                        }
                                    }
                                    
                                elif viz_subtype == "Box Plot":
                                    # For box plot, we need the actual data
                                    # Create a box plot grouped by master parameter
                                    fig = create_boxplot(df, selected_numeric, selected_master)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Store visualization for output
                                    st.session_state.visualization = {
                                        'type': 'master_box_plot',
                                        'figure': fig,
                                        'settings': {
                                            'master_column': selected_master,
                                            'numeric_column': selected_numeric
                                        }
                                    }
            
            elif st.session_state.visualization_type == "Histogram":
                st.subheader("Histogram")
                # Select column for histogram
                all_cols = st.session_state.selected_columns
                if all_cols:
                    hist_col = st.selectbox("Select column for histogram:", all_cols)
                    bins = st.slider("Number of bins:", min_value=5, max_value=100, value=20)
                    
                    # Create histogram
                    fig = create_histogram(df, hist_col, bins)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store visualization for output
                    st.session_state.visualization = {
                        'type': 'histogram',
                        'figure': fig,
                        'settings': {
                            'column': hist_col,
                            'bins': bins
                        }
                    }
                else:
                    st.warning("No columns available for histogram.")
            
            elif st.session_state.visualization_type == "Box Plot":
                st.subheader("Box Plot")
                # Select column for box plot
                all_cols = st.session_state.selected_columns
                if all_cols:
                    box_col = st.selectbox("Select column for box plot:", all_cols)
                    
                    # Optional grouping
                    group_by = None
                    use_grouping = st.checkbox("Group by another variable")
                    if use_grouping:
                        # Filter out the selected column
                        grouping_options = [col for col in all_cols if col != box_col]
                        if grouping_options:
                            group_by = st.selectbox("Select grouping variable:", grouping_options)
                    
                    # Create box plot
                    fig = create_boxplot(df, box_col, group_by)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store visualization for output
                    st.session_state.visualization = {
                        'type': 'boxplot',
                        'figure': fig,
                        'settings': {
                            'column': box_col,
                            'group_by': group_by
                        }
                    }
                else:
                    st.warning("No columns available for box plot.")
            
            elif st.session_state.visualization_type == "Scatter Plot":
                st.subheader("Scatter Plot")
                # Select columns for scatter plot
                all_cols = st.session_state.selected_columns
                if len(all_cols) >= 2:
                    x_col = st.selectbox("Select X-axis column:", all_cols, index=0)
                    # Filter out the X column from Y options
                    y_options = [col for col in all_cols if col != x_col]
                    y_col = st.selectbox("Select Y-axis column:", y_options, index=0)
                    
                    # Optional color grouping
                    color_by = None
                    # Filter out X and Y columns 
                    color_options = [col for col in all_cols if col != x_col and col != y_col]
                    if color_options:
                        use_color = st.checkbox("Color by another variable")
                        if use_color:
                            color_by = st.selectbox("Select color variable:", color_options)
                    
                    # Create scatter plot
                    fig = create_scatterplot(df, x_col, y_col, color_by)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store visualization for output
                    st.session_state.visualization = {
                        'type': 'scatterplot',
                        'figure': fig,
                        'settings': {
                            'x_col': x_col,
                            'y_col': y_col,
                            'color_by': color_by
                        }
                    }
                else:
                    st.warning("At least two columns are required for a scatter plot.")
            
            elif st.session_state.visualization_type == "Correlation Heatmap":
                st.subheader("Correlation Heatmap")
                # Select columns for correlation
                all_cols = st.session_state.selected_columns
                if len(all_cols) >= 2:
                    corr_cols = st.multiselect(
                        "Select columns for correlation heatmap:",
                        all_cols,
                        default=all_cols[:min(5, len(all_cols))]
                    )
                    
                    if len(corr_cols) >= 2:
                        corr_method = st.selectbox(
                            "Correlation method:",
                            ["pearson", "spearman", "kendall"]
                        )
                        
                        # Create correlation heatmap
                        fig = create_correlation_heatmap(df, corr_cols, corr_method)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store visualization for output
                        st.session_state.visualization = {
                            'type': 'correlation_heatmap',
                            'figure': fig,
                            'settings': {
                                'columns': corr_cols,
                                'method': corr_method
                            }
                        }
                    else:
                        st.info("Please select at least two columns for correlation heatmap.")
                else:
                    st.warning("At least two columns are required for a correlation heatmap.")
            
            elif st.session_state.visualization_type == "Distribution Plot":
                st.subheader("Distribution Plot")
                # Select column for distribution plot
                all_cols = st.session_state.selected_columns
                if all_cols:
                    dist_col = st.selectbox("Select column for distribution:", all_cols)
                    
                    # Create distribution plot
                    fig = create_distribution_plot(df, dist_col)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store visualization for output
                    st.session_state.visualization = {
                        'type': 'distribution_plot',
                        'figure': fig,
                        'settings': {
                            'column': dist_col
                        }
                    }
                else:
                    st.warning("No columns available for distribution plot.")
            
            elif st.session_state.visualization_type == "Bar Chart":
                st.subheader("Bar Chart")
                
                # Option for categorical data frequency or numeric summary by category
                chart_type = st.radio(
                    "Select bar chart type:",
                    ["Categorical Frequency", "Numeric Summary by Category"]
                )
                
                if chart_type == "Categorical Frequency":
                    cat_cols = [col for col in st.session_state.selected_columns if not pd.api.types.is_numeric_dtype(df[col])]
                    if cat_cols:
                        cat_col = st.selectbox("Select categorical column:", cat_cols)
                        # Get value counts and create a bar chart
                        value_counts = df[cat_col].value_counts().reset_index()
                        value_counts.columns = [cat_col, 'Count']
                        
                        # Limit display to top N categories if there are too many
                        top_n = st.slider("Show top N categories:", min_value=5, max_value=50, value=10)
                        if len(value_counts) > top_n:
                            value_counts = value_counts.head(top_n)
                            st.info(f"Showing top {top_n} categories only.")
                        
                        fig = px.bar(
                            value_counts, 
                            x=cat_col, 
                            y='Count',
                            title=f"Frequency of {cat_col} Categories",
                            labels={cat_col: cat_col, 'Count': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store visualization for output
                        st.session_state.visualization = {
                            'type': 'bar_chart_frequency',
                            'figure': fig,
                            'settings': {
                                'column': cat_col,
                                'top_n': top_n
                            }
                        }
                    else:
                        st.warning("No categorical columns available for frequency bar chart.")
                
                elif chart_type == "Numeric Summary by Category":
                    cat_cols = [col for col in st.session_state.selected_columns if not pd.api.types.is_numeric_dtype(df[col])]
                    num_cols = [col for col in st.session_state.selected_columns if pd.api.types.is_numeric_dtype(df[col])]
                    
                    if cat_cols and num_cols:
                        cat_col = st.selectbox("Select categorical column (X-axis):", cat_cols)
                        num_col = st.selectbox("Select numerical column to summarize:", num_cols)
                        
                        # Select summary statistic
                        summary_stat = st.selectbox(
                            "Select summary statistic:",
                            ["Mean", "Median", "Sum", "Min", "Max"]
                        )
                        
                        # Group by category and calculate the selected statistic
                        if summary_stat == "Mean":
                            summary_df = df.groupby(cat_col)[num_col].mean().reset_index()
                            stat_name = "Mean"
                        elif summary_stat == "Median":
                            summary_df = df.groupby(cat_col)[num_col].median().reset_index()
                            stat_name = "Median"
                        elif summary_stat == "Sum":
                            summary_df = df.groupby(cat_col)[num_col].sum().reset_index()
                            stat_name = "Sum"
                        elif summary_stat == "Min":
                            summary_df = df.groupby(cat_col)[num_col].min().reset_index()
                            stat_name = "Min"
                        elif summary_stat == "Max":
                            summary_df = df.groupby(cat_col)[num_col].max().reset_index()
                            stat_name = "Max"
                        
                        # Limit display to top N categories if there are too many
                        if len(summary_df) > 15:
                            top_n = st.slider("Show top N categories by value:", min_value=5, max_value=30, value=15)
                            summary_df = summary_df.sort_values(num_col, ascending=False).head(top_n)
                            st.info(f"Showing top {top_n} categories by {stat_name.lower()} value.")
                        
                        fig = px.bar(
                            summary_df, 
                            x=cat_col, 
                            y=num_col,
                            title=f"{stat_name} of {num_col} by {cat_col}",
                            labels={cat_col: cat_col, num_col: f"{stat_name} of {num_col}"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store visualization for output
                        st.session_state.visualization = {
                            'type': 'bar_chart_summary',
                            'figure': fig,
                            'settings': {
                                'cat_col': cat_col,
                                'num_col': num_col,
                                'summary_stat': summary_stat
                            }
                        }
                    else:
                        if not cat_cols:
                            st.warning("No categorical columns available for grouping.")
                        if not num_cols:
                            st.warning("No numerical columns available to summarize.")
            
            elif st.session_state.visualization_type == "Line Chart":
                st.subheader("Line Chart")
                
                # Check if there's any datetime column that could be used for a time series
                has_datetime = any(pd.api.types.is_datetime64_dtype(df[col]) for col in st.session_state.selected_columns)
                
                if has_datetime:
                    # Time series line chart
                    datetime_cols = [col for col in st.session_state.selected_columns if pd.api.types.is_datetime64_dtype(df[col])]
                    time_col = st.selectbox("Select time/date column (X-axis):", datetime_cols)
                    
                    other_cols = [col for col in st.session_state.selected_columns if col != time_col]
                    if other_cols:
                        y_cols = st.multiselect("Select column(s) to plot:", other_cols, default=[other_cols[0]])
                        
                        if y_cols:
                            # Create a copy of relevant data, sorted by time
                            plot_df = df[[time_col] + y_cols].sort_values(time_col).copy()
                            
                            # Create line chart
                            fig = px.line(
                                plot_df, 
                                x=time_col, 
                                y=y_cols,
                                title=f"Time Series of {', '.join(y_cols)}",
                                labels={col: col for col in [time_col] + y_cols}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Store visualization for output
                            st.session_state.visualization = {
                                'type': 'line_chart_time',
                                'figure': fig,
                                'settings': {
                                    'time_col': time_col,
                                    'y_cols': y_cols
                                }
                            }
                        else:
                            st.info("Please select at least one column to plot.")
                    else:
                        st.warning("No columns available for line chart.")
                
                else:
                    # Regular line chart (not time series)
                    all_cols = st.session_state.selected_columns
                    if len(all_cols) >= 2:
                        x_col = st.selectbox("Select X-axis column:", all_cols, index=0)
                        # Filter out the X column from Y options
                        y_options = [col for col in all_cols if col != x_col]
                        y_cols = st.multiselect("Select Y-axis column(s):", y_options, default=[y_options[0]])
                        
                        if y_cols:
                            # Create a copy of relevant data, sorted by X
                            plot_df = df[[x_col] + y_cols].sort_values(x_col).copy()
                            
                            # Create line chart
                            fig = px.line(
                                plot_df, 
                                x=x_col, 
                                y=y_cols,
                                title=f"Line Chart of {', '.join(y_cols)} vs {x_col}",
                                labels={col: col for col in [x_col] + y_cols}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Store visualization for output
                            st.session_state.visualization = {
                                'type': 'line_chart',
                                'figure': fig,
                                'settings': {
                                    'x_col': x_col,
                                    'y_cols': y_cols
                                }
                            }
                        else:
                            st.info("Please select at least one Y-axis column.")
                    else:
                        st.warning("At least two columns are required for a line chart.")
            
            # Visualization customization options
            with st.expander("Customize Visualization"):
                st.write("Customize your visualization with the following options:")
                
                # Title customization
                custom_title = st.text_input("Chart Title:", "")
                
                # Color scheme selection
                color_schemes = ["blues", "reds", "greens", "purples", "oranges", "viridis", "plasma", "inferno", "magma", "cividis"]
                color_scheme = st.selectbox("Color Scheme:", color_schemes)
                
                # Apply customizations if a visualization exists
                if 'visualization' in st.session_state and st.button("Apply Customizations"):
                    if 'figure' in st.session_state.visualization:
                        fig = st.session_state.visualization['figure']
                        
                        # Update title if provided
                        if custom_title:
                            fig.update_layout(title=custom_title)
                        
                        # Update color scheme
                        fig.update_layout(coloraxis=dict(colorscale=color_scheme))
                        
                        # Update the visualization in session state
                        st.session_state.visualization['figure'] = fig
                        st.session_state.visualization['customizations'] = {
                            'title': custom_title,
                            'color_scheme': color_scheme
                        }
                        
                        # Display the updated visualization
                        st.success("Customizations applied!")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Button to proceed to output
            if st.button("Continue to Output"):
                st.session_state.current_step = 6
                # Add JavaScript to scroll to top after rerun
                st.markdown(
                    """
                    <script>
                        window.scrollTo(0, 0);
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                st.rerun()
    
    elif st.session_state.current_step == 6:
        # Step 6: Output
        st.markdown('<div id="output"></div>', unsafe_allow_html=True)
        st.header("6. Output")
        
        if st.session_state.data is not None:
            st.subheader("Generate Analysis Report")
            
            # Options for report content
            include_summary = st.checkbox("Include Data Summary", value=True)
            include_stats = st.checkbox("Include Descriptive Statistics", value=True)
            include_viz = st.checkbox("Include Visualizations", value=True)
            include_corr = st.checkbox("Include Correlation Analysis", value=True)
            
            if st.button("Generate HTML Report"):
                try:
                    # Generate the HTML report
                    html_report = generate_report(
                        st.session_state.data[st.session_state.selected_columns] if st.session_state.selected_columns else st.session_state.data,
                        filename=st.session_state.filename,
                        include_summary=include_summary,
                        include_stats=include_stats,
                        include_viz=include_viz,
                        include_corr=include_corr
                    )
                    
                    # Create a download link
                    b64 = base64.b64encode(html_report.encode("utf-8")).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="statistical_analysis_report.html">Download HTML Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Show a preview
                    with st.expander("Preview Report"):
                        st.components.v1.html(html_report, height=500, scrolling=True)
                    
                    st.success("Report generated successfully! Click the link above to download.")
                
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
            
            # Export data option
            st.subheader("Export Analysis Data")
            
            export_format = st.selectbox(
                "Select export format:",
                ["CSV", "Excel"]
            )
            
            if st.button("Export Data"):
                try:
                    export_df = st.session_state.data[st.session_state.selected_columns] if st.session_state.selected_columns else st.session_state.data
                    
                    if export_format == "CSV":
                        csv = export_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_data.csv">Download CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    elif export_format == "Excel":
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            export_df.to_excel(writer, index=False, sheet_name='Data')
                            
                            # Add a sheet for statistics if available
                            if 'analysis_results' in st.session_state:
                                result = st.session_state.analysis_results
                                if result['type'] == 'descriptive' and 'stats_df' in result:
                                    result['stats_df'].to_excel(writer, sheet_name='Statistics')
                                elif result['type'] in ['ttest', 'anova', 'correlation'] and 'result' in result:
                                    # Create a summary sheet with the results
                                    summary_df = pd.DataFrame({'Key': ['Analysis Type'], 'Value': [result['type']]})
                                    summary_df.to_excel(writer, sheet_name='Analysis Results', index=False)
                        
                        b64 = base64.b64encode(output.getvalue()).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="analysis_data.xlsx">Download Excel</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    st.success(f"Data exported successfully as {export_format}! Click the link above to download.")
                
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")
            
            # Save visualization option if available
            if 'visualization' in st.session_state and 'figure' in st.session_state.visualization:
                st.subheader("Export Visualization")
                
                viz_format = st.selectbox(
                    "Select image format:",
                    ["PNG", "JPEG", "SVG", "PDF"]
                )
                
                if st.button("Export Visualization"):
                    try:
                        fig = st.session_state.visualization['figure']
                        
                        # Get the image bytes in the selected format
                        if viz_format == "PNG":
                            img_bytes = fig.to_image(format="png")
                            mime_type = "image/png"
                            file_ext = "png"
                        elif viz_format == "JPEG":
                            img_bytes = fig.to_image(format="jpeg")
                            mime_type = "image/jpeg"
                            file_ext = "jpg"
                        elif viz_format == "SVG":
                            img_bytes = fig.to_image(format="svg")
                            mime_type = "image/svg+xml"
                            file_ext = "svg"
                        elif viz_format == "PDF":
                            img_bytes = fig.to_image(format="pdf")
                            mime_type = "application/pdf"
                            file_ext = "pdf"
                        
                        # Create a download link
                        b64 = base64.b64encode(img_bytes).decode()
                        href = f'<a href="data:{mime_type};base64,{b64}" download="visualization.{file_ext}">Download {viz_format}</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        st.success(f"Visualization exported as {viz_format}! Click the link above to download.")
                    
                    except Exception as e:
                        st.error(f"Error exporting visualization: {str(e)}")
            
            # Start over option
            st.subheader("Start a New Analysis")
            if st.button("Start Over"):
                # Reset session state
                st.session_state.data = None
                st.session_state.filename = None
                st.session_state.current_step = 1
                st.session_state.selected_columns = []
                st.session_state.analysis_type = None
                st.session_state.visualization_type = None
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
                if 'visualization' in st.session_state:
                    del st.session_state.visualization
                if 'ttest_settings' in st.session_state:
                    del st.session_state.ttest_settings
                if 'anova_settings' in st.session_state:
                    del st.session_state.anova_settings
                if 'corr_settings' in st.session_state:
                    del st.session_state.corr_settings
                
                st.success("Session reset. Ready for a new analysis!")
                st.rerun()
        
        # Tab 1: Data Exploration
        with tab1:
            st.header("Data Exploration")
            
            # Display full dataframe with pagination
            st.subheader("Full Dataset")
            st.dataframe(df)
            
            # Display column information
            st.subheader("Column Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Numerical Columns:")
                numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
                if numerical_cols:
                    st.write(", ".join(numerical_cols))
                else:
                    st.write("No numerical columns found")
            
            with col2:
                st.write("Categorical Columns:")
                categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                if categorical_cols:
                    st.write(", ".join(categorical_cols))
                else:
                    st.write("No categorical columns found")
            
            # Check for missing values
            st.subheader("Missing Values")
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                st.write("Columns with missing values:")
                missing_df = pd.DataFrame({
                    'Column': missing_values.index,
                    'Missing Values': missing_values.values,
                    'Percentage': (missing_values.values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
                st.table(missing_df)
            else:
                st.write("No missing values found in the dataset.")
            
            # Data types
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values
            })
            st.table(dtypes_df)
            
            # Age Grouping (now handled in drag_drop_ui.py after clicking Complete Configuration)
            st.subheader("Age Grouping")
            
            # Display information about age group generation
            if 'age_column' in st.session_state and st.session_state.age_column:
                st.info("Age groups will be generated based on your selected age column after clicking 'Complete Configuration' in the Drag & Drop interface.")
            else:
                st.info("To generate age groups, please assign an age column in the Drag & Drop interface.")
            
            # Check for our generated age group column from session state first
            age_group_col = None
            if 'generated_age_group_col' in st.session_state and st.session_state.generated_age_group_col in df.columns:
                # Use our explicitly generated age group column from drag & drop interface
                age_group_col = st.session_state.generated_age_group_col
                st.success(f"Using the age groups generated from your configuration: '{age_group_col}'")
            else:
                # Fall back to checking for existing columns in the dataframe
                existing_age_group_cols = [col for col in df.columns if 'age group' in col.lower() or 'Age group' in col]
                generated_age_group_cols = [col for col in df.columns if 'Generated Age Group' in col]
                
                # Combine both types of age group columns
                all_age_group_cols = existing_age_group_cols + generated_age_group_cols
                
                if all_age_group_cols:
                    # Still prioritize any column named 'Generated Age Group' if it exists
                    if 'Generated Age Group' in all_age_group_cols:
                        age_group_col = 'Generated Age Group'
                    else:
                        age_group_col = all_age_group_cols[0]
                        st.info(f"Using existing age group column: '{age_group_col}' from the Excel file. To create custom age groups, assign an age column in the configuration.")
            
            # Display age group distribution if we have an age column and an age group column
            if age_group_col is not None and 'age_column' in st.session_state and st.session_state.age_column:
                age_column = st.session_state.age_column['name']
                age_type = st.session_state.age_type if 'age_type' in st.session_state else "years"
                
                st.info(f"Age groups have been created based on {age_column} (measured in {age_type}).")
                
                # Display age group distribution
                age_group_counts = df[age_group_col].value_counts().sort_index()
                st.write("**Age Group Distribution:**")
                st.bar_chart(age_group_counts)
                
                # Show dataframe with the age group column next to age column
                st.write("**Dataset with Age Groups:**")
                st.dataframe(df[[age_column, age_group_col]].head())
                
                # Option to download with age groups
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Data with Age Groups",
                    csv,
                    "data_with_age_groups.csv",
                    "text/csv",
                    key='download-csv-age-groups'
                )
            elif age_group_col is None:
                st.info("Age groups will be created when you click 'Complete Configuration' in the Drag & Drop interface.")
            
        # Tab 2: Descriptive Statistics
        with tab2:
            st.header("Descriptive Statistics")
            
            # Calculate statistics for ALL columns by default
            all_cols = df.columns.tolist()
            
            # Still allow selection for display, but default to all columns
            selected_cols = st.multiselect(
                "Select columns for analysis (defaults to all columns):", 
                all_cols,
                default=all_cols
            )
            
            if selected_cols:
                try:
                    # Generate and display descriptive statistics
                    stats_df = get_descriptive_stats(df, selected_cols)
                    
                    # Convert 'Mean ± Std Dev' to a string column to avoid Arrow conversion errors
                    for col in stats_df.columns:
                        if isinstance(stats_df.loc['Mean ± Std Dev', col], str):
                            # If Mean ± Std Dev is already a string, we need to convert the whole column to strings
                            # for proper display in the dataframe
                            stats_df[col] = stats_df[col].astype(str)
                    
                    st.subheader("Summary Statistics")
                    st.dataframe(stats_df)
                    
                    # Display additional statistics for all columns
                    # Calculate statistics for all columns
                    st.subheader("Additional Statistics for All Columns")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Skewness:")
                        # Create DataFrame with all columns
                        skew_data = []
                        for col in selected_cols:
                            try:
                                # Try to calculate skewness - will work for numerical and some other types
                                skew_value = df[col].skew()
                                skew_data.append({'Column': col, 'Skewness': skew_value})
                            except:
                                # For columns where skewness can't be calculated
                                skew_data.append({'Column': col, 'Skewness': "NA"})
                        
                        skew_df = pd.DataFrame(skew_data)
                        st.table(skew_df)
                        
                        with col2:
                            st.write("Kurtosis:")
                            kurt_df = pd.DataFrame({
                                'Column': numerical_selected,
                                'Kurtosis': [df[col].kurtosis() for col in numerical_selected]
                            })
                            st.table(kurt_df)
                    
                    # Display frequency tables for categorical columns
                    categorical_selected = [col for col in selected_cols if not pd.api.types.is_numeric_dtype(df[col])]
                    if categorical_selected:
                        st.subheader("Frequency Tables for Categorical Columns")
                        for col in categorical_selected:
                            st.write(f"**{col}:**")
                            freq_table = df[col].value_counts().reset_index()
                            freq_table.columns = [col, 'Count']
                            freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
                            st.table(freq_table)
                
                except Exception as e:
                    st.error(f"Error calculating statistics: {str(e)}")
            else:
                st.info("Please select at least one column for analysis.")
        
        # Tab 3: Visualizations
        with tab3:
            st.header("Data Visualization")
            
            # Visualization type selector
            viz_type = st.selectbox(
                "Select visualization type:",
                ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap", "Distribution Plot"]
            )
            
            if viz_type == "Histogram":
                st.subheader("Histogram")
                # Select column for histogram
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    hist_col = st.selectbox("Select column for histogram:", num_cols)
                    bins = st.slider("Number of bins:", min_value=5, max_value=100, value=20)
                    
                    # Create histogram
                    fig = create_histogram(df, hist_col, bins)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numerical columns available for histogram.")
            
            elif viz_type == "Box Plot":
                st.subheader("Box Plot")
                # Select column for box plot
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    box_col = st.selectbox("Select numerical column for box plot:", num_cols)
                    
                    # Optional grouping
                    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                    group_by = None
                    if cat_cols:
                        use_grouping = st.checkbox("Group by categorical variable")
                        if use_grouping:
                            group_by = st.selectbox("Select grouping variable:", cat_cols)
                    
                    # Create box plot
                    fig = create_boxplot(df, box_col, group_by)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numerical columns available for box plot.")
            
            elif viz_type == "Scatter Plot":
                st.subheader("Scatter Plot")
                # Select columns for scatter plot
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) >= 2:
                    x_col = st.selectbox("Select X-axis column:", num_cols, index=0)
                    y_col = st.selectbox("Select Y-axis column:", num_cols, index=min(1, len(num_cols)-1))
                    
                    # Optional color grouping
                    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                    color_by = None
                    if cat_cols:
                        use_color = st.checkbox("Color by categorical variable")
                        if use_color:
                            color_by = st.selectbox("Select color variable:", cat_cols)
                    
                    # Create scatter plot
                    fig = create_scatterplot(df, x_col, y_col, color_by)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("At least two numerical columns are required for a scatter plot.")
            
            elif viz_type == "Correlation Heatmap":
                st.subheader("Correlation Heatmap")
                # Select columns for correlation
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) >= 2:
                    corr_cols = st.multiselect(
                        "Select columns for correlation analysis:",
                        num_cols,
                        default=num_cols[:min(5, len(num_cols))]
                    )
                    
                    if len(corr_cols) >= 2:
                        corr_method = st.selectbox(
                            "Correlation method:",
                            ["pearson", "spearman", "kendall"]
                        )
                        
                        # Create correlation heatmap
                        fig = create_correlation_heatmap(df, corr_cols, corr_method)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least two columns for correlation analysis.")
                else:
                    st.warning("At least two numerical columns are required for a correlation heatmap.")
            
            elif viz_type == "Distribution Plot":
                st.subheader("Distribution Plot")
                # Select column for distribution plot
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    dist_col = st.selectbox("Select column for distribution:", num_cols)
                    
                    # Create distribution plot
                    fig = create_distribution_plot(df, dist_col)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numerical columns available for distribution plot.")
        
        # Tab 4: Statistical Tests
        with tab4:
            st.header("Statistical Tests")
            
            test_type = st.selectbox(
                "Select statistical test:",
                ["t-Test", "ANOVA", "Correlation Analysis"]
            )
            
            if test_type == "t-Test":
                st.subheader("t-Test")
                
                # Get all columns for selection
                all_cols = df.columns.tolist()
                
                # Also keep track of numerical columns for validation
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                
                if len(all_cols) > 0:
                    # Test type
                    ttest_type = st.radio(
                        "Select t-test type:",
                        ["One-sample t-test", "Two-sample t-test"]
                    )
                    
                    if ttest_type == "One-sample t-test":
                        col = st.selectbox("Select column for test:", all_cols)
                        mu = st.number_input("Population mean (μ₀):", value=0.0)
                        alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05)
                        
                        if st.button("Perform One-sample t-test"):
                            try:
                                result = perform_ttest(df, col, test_type="one-sample", mu=mu, alpha=alpha)
                                
                                # Display results
                                st.write("### Results")
                                st.write(f"Sample mean: {result['sample_mean']:.4f}")
                                st.write(f"t-statistic: {result['t_stat']:.4f}")
                                st.write(f"p-value: {result['p_value']:.4f}")
                                st.write(f"Degrees of freedom: {result['df']}")
                                
                                # Interpretation
                                st.write("### Interpretation")
                                if result['p_value'] < alpha:
                                    st.write(f"With a p-value of {result['p_value']:.4f}, which is less than the significance level of {alpha}, we reject the null hypothesis.")
                                    st.write(f"This suggests that the true mean is significantly different from {mu}.")
                                else:
                                    st.write(f"With a p-value of {result['p_value']:.4f}, which is greater than the significance level of {alpha}, we fail to reject the null hypothesis.")
                                    st.write(f"This suggests that there is not enough evidence to conclude that the true mean is different from {mu}.")
                                
                                # Confidence interval
                                st.write(f"95% Confidence Interval: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")
                            
                            except Exception as e:
                                st.error(f"Error performing t-test: {str(e)}")
                    
                    elif ttest_type == "Two-sample t-test":
                        # Two options: compare two numerical columns OR compare one numerical column grouped by categories
                        comparison_type = st.radio(
                            "Select comparison method:",
                            ["Compare two columns", "Compare groups within a column"]
                        )
                        
                        if comparison_type == "Compare two columns":
                            col1 = st.selectbox("Select first column:", all_cols, index=0)
                            col2 = st.selectbox("Select second column:", all_cols, index=min(1, len(all_cols)-1))
                            equal_var = st.checkbox("Assume equal variance", value=False)
                            alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05)
                            
                            if st.button("Perform Two-sample t-test (columns)"):
                                try:
                                    result = perform_ttest(df, col1, col2=col2, test_type="two-sample-cols", 
                                                        equal_var=equal_var, alpha=alpha)
                                    
                                    # Display results
                                    st.write("### Results")
                                    st.write(f"Mean of {col1}: {result['mean1']:.4f}")
                                    st.write(f"Mean of {col2}: {result['mean2']:.4f}")
                                    st.write(f"Mean difference: {result['mean_diff']:.4f}")
                                    st.write(f"t-statistic: {result['t_stat']:.4f}")
                                    st.write(f"p-value: {result['p_value']:.4f}")
                                    st.write(f"Degrees of freedom: {result['df']:.1f}")
                                    
                                    # Interpretation
                                    st.write("### Interpretation")
                                    if result['p_value'] < alpha:
                                        st.write(f"With a p-value of {result['p_value']:.4f}, which is less than the significance level of {alpha}, we reject the null hypothesis.")
                                        st.write(f"This suggests that there is a significant difference between the means of {col1} and {col2}.")
                                    else:
                                        st.write(f"With a p-value of {result['p_value']:.4f}, which is greater than the significance level of {alpha}, we fail to reject the null hypothesis.")
                                        st.write(f"This suggests that there is not enough evidence to conclude that the means of {col1} and {col2} are significantly different.")
                                    
                                    # Confidence interval
                                    st.write(f"95% Confidence Interval of the difference: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")
                                
                                except Exception as e:
                                    st.error(f"Error performing t-test: {str(e)}")
                        
                        else:  # Compare groups within a column
                            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                            if not cat_cols:
                                st.warning("No categorical columns available for grouping.")
                            else:
                                num_col = st.selectbox("Select column to test:", all_cols)
                                group_col = st.selectbox("Select categorical column for grouping:", cat_cols)
                                
                                # Get unique values in the categorical column
                                unique_vals = df[group_col].unique()
                                if len(unique_vals) < 2:
                                    st.warning(f"Column '{group_col}' must have at least 2 unique values for comparison.")
                                else:
                                    if len(unique_vals) > 2:
                                        st.info(f"Column '{group_col}' has {len(unique_vals)} unique values. t-test requires exactly 2 groups.")
                                    
                                    group1 = st.selectbox("Select first group:", unique_vals, index=0)
                                    group2 = st.selectbox("Select second group:", unique_vals, index=min(1, len(unique_vals)-1))
                                    
                                    equal_var = st.checkbox("Assume equal variance", value=False)
                                    alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05)
                                    
                                    if st.button("Perform Two-sample t-test (groups)"):
                                        try:
                                            result = perform_ttest(df, num_col, test_type="two-sample-groups",
                                                                group_col=group_col, group1=group1, group2=group2,
                                                                equal_var=equal_var, alpha=alpha)
                                            
                                            # Display results
                                            st.write("### Results")
                                            st.write(f"Mean of {num_col} for {group_col}={group1}: {result['mean1']:.4f}")
                                            st.write(f"Mean of {num_col} for {group_col}={group2}: {result['mean2']:.4f}")
                                            st.write(f"Mean difference: {result['mean_diff']:.4f}")
                                            st.write(f"t-statistic: {result['t_stat']:.4f}")
                                            st.write(f"p-value: {result['p_value']:.4f}")
                                            st.write(f"Degrees of freedom: {result['df']:.1f}")
                                            
                                            # Interpretation
                                            st.write("### Interpretation")
                                            if result['p_value'] < alpha:
                                                st.write(f"With a p-value of {result['p_value']:.4f}, which is less than the significance level of {alpha}, we reject the null hypothesis.")
                                                st.write(f"This suggests that there is a significant difference in {num_col} between {group_col}={group1} and {group_col}={group2}.")
                                            else:
                                                st.write(f"With a p-value of {result['p_value']:.4f}, which is greater than the significance level of {alpha}, we fail to reject the null hypothesis.")
                                                st.write(f"This suggests that there is not enough evidence to conclude that {num_col} differs significantly between {group_col}={group1} and {group_col}={group2}.")
                                            
                                            # Confidence interval
                                            st.write(f"95% Confidence Interval of the difference: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")
                                        
                                        except Exception as e:
                                            st.error(f"Error performing t-test: {str(e)}")
                else:
                    st.warning("No numerical columns available for t-test.")
            
            elif test_type == "ANOVA":
                st.subheader("Analysis of Variance (ANOVA)")
                
                # Get all columns for selection
                all_cols = df.columns.tolist()
                
                # Need at least one numerical column (dependent variable) and one categorical column (factor)
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                
                if not num_cols:
                    st.warning("No numerical columns available for ANOVA.")
                elif not cat_cols:
                    st.warning("No categorical columns available for ANOVA.")
                else:
                    # Allow any column type for selection, not just numerical
                    num_col = st.selectbox("Select column (dependent variable):", all_cols)
                    cat_col = st.selectbox("Select categorical column (factor):", cat_cols)
                    
                    # Check the number of groups in the categorical column
                    unique_groups = df[cat_col].nunique()
                    if unique_groups < 2:
                        st.warning(f"Column '{cat_col}' must have at least 2 groups for ANOVA.")
                    else:
                        alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05)
                        
                        if st.button("Perform ANOVA"):
                            try:
                                result = perform_anova(df, num_col, cat_col, alpha)
                                
                                # Display results
                                st.write("### Results")
                                st.write(f"F-statistic: {result['f_stat']:.4f}")
                                st.write(f"p-value: {result['p_value']:.4f}")
                                
                                # ANOVA table
                                st.write("### ANOVA Table")
                                st.dataframe(result['anova_table'])
                                
                                # Group statistics
                                st.write("### Group Statistics")
                                st.dataframe(result['group_stats'])
                                
                                # Interpretation
                                st.write("### Interpretation")
                                if result['p_value'] < alpha:
                                    st.write(f"With a p-value of {result['p_value']:.4f}, which is less than the significance level of {alpha}, we reject the null hypothesis.")
                                    st.write(f"This suggests that at least one group in {cat_col} has a significantly different mean {num_col} from the others.")
                                else:
                                    st.write(f"With a p-value of {result['p_value']:.4f}, which is greater than the significance level of {alpha}, we fail to reject the null hypothesis.")
                                    st.write(f"This suggests that there is not enough evidence to conclude that any group in {cat_col} has a significantly different mean {num_col} from the others.")
                                
                                # Visualization of group means
                                st.write("### Visualization of Group Means")
                                fig = create_boxplot(df, num_col, cat_col)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            except Exception as e:
                                st.error(f"Error performing ANOVA: {str(e)}")
            
            elif test_type == "Correlation Analysis":
                st.subheader("Correlation Analysis")
                
                # Get all columns for selection
                all_cols = df.columns.tolist()
                
                # Also keep track of numerical columns for default selection
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                
                if len(num_cols) < 2:
                    st.warning("At least two numerical columns are recommended for correlation analysis.")
                
                # Allow any column for selection, not just numerical
                corr_cols = st.multiselect(
                    "Select columns for correlation analysis:",
                    all_cols,
                    default=num_cols[:min(3, len(num_cols))] if num_cols else all_cols[:min(3, len(all_cols))]
                )
                
                if len(corr_cols) < 2:
                    st.info("Please select at least two columns for correlation analysis.")
                else:
                    corr_method = st.selectbox(
                        "Correlation method:",
                        ["pearson", "spearman", "kendall"]
                    )
                    alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05)
                    
                    if st.button("Perform Correlation Analysis"):
                        try:
                            result = perform_correlation_analysis(df, corr_cols, method=corr_method, alpha=alpha)
                            
                            # Display correlation matrix
                            st.write("### Correlation Matrix")
                            st.dataframe(result['corr_matrix'])
                            
                            # Display p-values matrix
                            st.write("### P-values Matrix")
                            st.dataframe(result['p_matrix'])
                            
                            # Display correlation heatmap
                            st.write("### Correlation Heatmap")
                            fig = create_correlation_heatmap(df, corr_cols, corr_method)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display significant correlations
                            st.write("### Significant Correlations")
                            if not result['significant_corrs'].empty:
                                st.dataframe(result['significant_corrs'])
                            else:
                                st.write("No significant correlations found at α = {}.".format(alpha))
                            
                            # Scatter plot matrix
                            st.write("### Scatter Plot Matrix")
                            st.write("Generating scatter plot matrix for selected variables...")
                            for i in range(len(corr_cols)):
                                for j in range(i+1, len(corr_cols)):
                                    x_col = corr_cols[i]
                                    y_col = corr_cols[j]
                                    st.write(f"**{x_col}** vs **{y_col}**")
                                    st.write(f"Correlation ({corr_method}): {result['corr_matrix'].loc[x_col, y_col]:.4f}, p-value: {result['p_matrix'].loc[x_col, y_col]:.4f}")
                                    fig = create_scatterplot(df, x_col, y_col)
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error performing correlation analysis: {str(e)}")
        
        # Tab 5: Export Results
        with tab5:
            st.header("Export Results")
            
            export_type = st.selectbox(
                "Select export format:",
                ["CSV", "Excel", "HTML Report", "JSON"]
            )
            
            if export_type == "CSV":
                st.subheader("Export to CSV")
                
                export_options = st.multiselect(
                    "Select data to export:",
                    ["Original Data", "Descriptive Statistics"]
                )
                
                if "Original Data" in export_options:
                    # Generate CSV download link for original data
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{st.session_state.filename.split(".")[0]}_data.csv">Download Original Data as CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                if "Descriptive Statistics" in export_options:
                    # Generate statistics for all columns, not just numerical ones
                    all_cols = df.columns.tolist()
                    if all_cols:
                        stats_df = get_descriptive_stats(df, all_cols)
                        
                        # Convert any problematic values to strings for proper CSV export
                        for col in stats_df.columns:
                            if 'Mean ± Std Dev' in stats_df.index and isinstance(stats_df.loc['Mean ± Std Dev', col], str):
                                stats_df[col] = stats_df[col].astype(str)
                                
                        csv = stats_df.to_csv(index=True)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="{st.session_state.filename.split(".")[0]}_stats.csv">Download Descriptive Statistics as CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.warning("No columns available for statistics.")
            
            elif export_type == "Excel":
                st.subheader("Export to Excel")
                
                export_options = st.multiselect(
                    "Select data to export:",
                    ["Original Data", "Descriptive Statistics"]
                )
                
                if export_options:
                    # Create a BytesIO object to hold the Excel file
                    output = io.BytesIO()
                    
                    # Create Excel writer
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        if "Original Data" in export_options:
                            df.to_excel(writer, sheet_name='Data', index=False)
                        
                        if "Descriptive Statistics" in export_options:
                            # Get all columns, not just numerical ones
                            all_cols = df.columns.tolist()
                            if all_cols:
                                stats_df = get_descriptive_stats(df, all_cols)
                                
                                # Convert 'Mean ± Std Dev' to a string column to avoid Arrow conversion errors
                                for col in stats_df.columns:
                                    if isinstance(stats_df.loc['Mean ± Std Dev', col], str):
                                        # If Mean ± Std Dev is already a string, we need to convert the whole column to strings
                                        stats_df[col] = stats_df[col].astype(str)
                                
                                stats_df.to_excel(writer, sheet_name='Statistics')
                    
                    # Create download link
                    output.seek(0)
                    b64 = base64.b64encode(output.read()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{st.session_state.filename.split(".")[0]}_analysis.xlsx">Download Excel file</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            elif export_type == "HTML Report":
                st.subheader("Generate HTML Report")
                
                include_sections = st.multiselect(
                    "Select sections to include in the report:",
                    ["Data Summary", "Descriptive Statistics", "Visualizations", "Correlation Analysis"]
                )
                
                if include_sections:
                    # Generate HTML report
                    if st.button("Generate HTML Report"):
                        report_html = generate_report(
                            df, 
                            filename=st.session_state.filename,
                            include_summary="Data Summary" in include_sections,
                            include_stats="Descriptive Statistics" in include_sections,
                            include_viz="Visualizations" in include_sections,
                            include_corr="Correlation Analysis" in include_sections
                        )
                        
                        # Create download link
                        b64 = base64.b64encode(report_html.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="{st.session_state.filename.split(".")[0]}_report.html">Download HTML Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        # Preview
                        with st.expander("Preview Report"):
                            st.components.v1.html(report_html, height=600, scrolling=True)
            
            elif export_type == "JSON":
                st.subheader("Export to JSON")
                
                export_options = st.multiselect(
                    "Select data to export:",
                    ["Original Data", "Descriptive Statistics"]
                )
                
                if "Original Data" in export_options:
                    # Generate JSON download link for original data
                    json_str = df.to_json(orient="records", date_format="iso")
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="{st.session_state.filename.split(".")[0]}_data.json">Download Original Data as JSON</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                if "Descriptive Statistics" in export_options:
                    # Generate statistics for all columns, not just numerical ones
                    all_cols = df.columns.tolist()
                    if all_cols:
                        stats_df = get_descriptive_stats(df, all_cols)
                        
                        # Convert 'Mean ± Std Dev' to a string column to avoid Arrow conversion errors
                        for col in stats_df.columns:
                            if isinstance(stats_df.loc['Mean ± Std Dev', col], str):
                                # If Mean ± Std Dev is already a string, we need to convert the whole column to strings
                                stats_df[col] = stats_df[col].astype(str)
                        
                        # Convert to JSON with required formatting
                        json_str = stats_df.to_json(orient="split")
                        b64 = base64.b64encode(json_str.encode()).decode()
                        href = f'<a href="data:application/json;base64,{b64}" download="{st.session_state.filename.split(".")[0]}_stats.json">Download Descriptive Statistics as JSON</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.warning("No columns available for statistics.")

    else:
        # Display a message when no data is loaded
        st.info("Please upload a data file or use the sample dataset to get started.")

if __name__ == "__main__":
    main()
