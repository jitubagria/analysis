import streamlit as st
import pandas as pd
import json

# Initialize the session state
def initialize_state():
    """Initialize the session state for configuration"""
    if 'master_columns' not in st.session_state:
        st.session_state.master_columns = []
    if 'age_column' not in st.session_state:
        st.session_state.age_column = None  
    if 'age_type' not in st.session_state:
        st.session_state.age_type = "years"
    if 'gender_column' not in st.session_state:
        st.session_state.gender_column = None
    if 'pair_areas' not in st.session_state:
        st.session_state.pair_areas = [{'id': 'pair-1', 'pre': [], 'post': []}]
    if 'pair_count' not in st.session_state:
        st.session_state.pair_count = 1
    if 'configuration_complete' not in st.session_state:
        st.session_state.configuration_complete = False

def display_drag_drop_ui():
    """Display a user-friendly column configuration interface"""
    # Title is now provided by the parent app.py instead of here
    
    # Initialize session state
    initialize_state()
    
    # Check if we have data loaded
    if 'data' in st.session_state and st.session_state.data is not None:
        df = st.session_state.data
        
        # Column options for selection
        all_columns = df.columns.tolist()
        
        # Master Area (Optional)
        st.subheader("Master Area (Optional)")
        st.write("These columns will be available for all analyses.")
        master_cols = st.multiselect(
            "Select columns for Master Area:",
            all_columns
        )
        
        # Age Field (Optional) 
        st.subheader("Age Field (Optional)")
        age_col = st.selectbox(
            "Select Age column:",
            ["None"] + all_columns
        )
        
        if age_col != "None":
            age_type = st.radio(
                "Age Type:",
                ["Years", "Months", "Days"],
                horizontal=True
            )
            st.session_state.age_type = age_type.lower()
        
        # Gender Field (Optional)
        st.subheader("Gender Field (Optional)")
        gender_col = st.selectbox(
            "Select Gender column:",
            ["None"] + all_columns
        )
        
        # Pairs (Optional)
        st.subheader("Pre/Post Pairs (Optional)")
        
        # Use tabs for each pair
        pair_tabs = []
        for i in range(st.session_state.pair_count):
            pair_tabs.append(f"Pair {i+1}")
        
        tab_index = st.tabs(pair_tabs)
        
        pairs = []
        for i, tab in enumerate(tab_index):
            with tab:
                st.write(f"#### Pair {i+1}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Pre")
                    pre_cols = st.multiselect(
                        f"Pre columns for Pair {i+1}:",
                        all_columns,
                        key=f"pre_{i}"
                    )
                
                with col2:
                    st.write("Post")
                    post_cols = st.multiselect(
                        f"Post columns for Pair {i+1}:",
                        all_columns,
                        key=f"post_{i}"
                    )
                
                pairs.append({
                    "id": f"pair-{i+1}",
                    "pre": [{"id": f"col_{j}", "name": col} for j, col in enumerate(pre_cols)],
                    "post": [{"id": f"col_{j}", "name": col} for j, col in enumerate(post_cols)]
                })
                
                if i > 0:
                    if st.button(f"Remove Pair {i+1}", key=f"remove_pair_{i}"):
                        st.session_state.pair_count -= 1
                        st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Pair"):
                st.session_state.pair_count += 1
                st.rerun()
        
        # Save configuration button
        if st.button("Complete Configuration"):
            # Update session state variables based on selections
            st.session_state.master_columns = [{"id": f"col_{i}", "name": col} for i, col in enumerate(master_cols)]
            st.write(f"Debug: Master columns set to: {master_cols}")
            
            if age_col != "None":
                st.session_state.age_column = {"id": f"age_field", "name": age_col}
            else:
                st.session_state.age_column = None
                
            if gender_col != "None":
                st.session_state.gender_column = {"id": f"gender_field", "name": gender_col}
            else:
                st.session_state.gender_column = None
                
            st.session_state.pair_areas = pairs
            st.session_state.configuration_complete = True
            
            # Now that configuration is complete, generate age groups if an age column was selected
            if st.session_state.age_column is not None:
                from data_utils import generate_age_groups
                import numpy as np
                
                age_col = st.session_state.age_column['name']
                age_type = st.session_state.age_type
                
                # Use a consistent name for our generated age group
                GENERATED_AGE_GROUP_NAME = 'Generated Age Group'
                
                try:
                    # Generate age groups using our function
                    df_with_groups, grouping_info, age_group_col = generate_age_groups(df, age_col, age_type, GENERATED_AGE_GROUP_NAME)
                    
                    # Store the name of the generated age group column in session state for later reference
                    st.session_state.generated_age_group_col = age_group_col
                    
                    # Update the dataframe with age groups
                    df = df_with_groups
                    st.session_state.data = df_with_groups
                    
                    # Also store the grouping info in session state for analysis
                    st.session_state.age_grouping_info = grouping_info
                    
                    st.success(f"Age groups created based on {age_col} (measured in {age_type}).")
                except Exception as e:
                    st.warning(f"Could not create age groups: {str(e)}")
            
            st.success("Configuration completed successfully!")
            
            # Display the final dataset analysis after successful configuration
            st.header("Final Dataset for Analysis")
            
            # Determine display columns based on the configuration
            display_columns = []
            
            # Add master columns
            if st.session_state.master_columns:
                for col in st.session_state.master_columns:
                    if col['name'] not in display_columns:
                        display_columns.append(col['name'])
            
            # Add age column and age group if available
            if st.session_state.age_column:
                age_col = st.session_state.age_column['name']
                if age_col not in display_columns:
                    display_columns.append(age_col)
                
                # Check for our generated age group column first
                if 'generated_age_group_col' in st.session_state:
                    # If we've generated our own age group column, use it for analysis
                    generated_col = st.session_state.generated_age_group_col
                    if generated_col in df.columns and generated_col not in display_columns:
                        display_columns.append(generated_col)
                else:
                    # Only if we didn't generate our own, check for existing ones
                    existing_age_group_cols = [col for col in df.columns if 'age group' in col.lower() or 'Age group' in col]
                    if existing_age_group_cols and existing_age_group_cols[0] not in display_columns:
                        display_columns.append(existing_age_group_cols[0])
                        st.info("Using existing age group column from the Excel file. For custom age groups, please assign an age column and click 'Complete Configuration'.")
            
            # Add gender column
            if st.session_state.gender_column:
                gender_col = st.session_state.gender_column['name']
                if gender_col not in display_columns:
                    display_columns.append(gender_col)
            
            # Add pair columns
            if st.session_state.pair_areas:
                for pair in st.session_state.pair_areas:
                    for area_type in ['pre', 'post']:
                        for col in pair[area_type]:
                            if col['name'] not in display_columns:
                                display_columns.append(col['name'])
            
            # Create a styled dataframe with background colors for different column types
            if display_columns:
                # Filter to show only columns that exist in the dataframe
                existing_display_columns = [col for col in display_columns if col in df.columns]
                
                if existing_display_columns:
                    # Show all rows in the dataset instead of just 10
                    styled_df = df[existing_display_columns].style
                    
                    # Apply background colors based on column types
                    def highlight_columns(x):
                        styles = pd.DataFrame('', index=x.index, columns=x.columns)
                        
                        # Master columns (light blue)
                        if st.session_state.master_columns:
                            master_cols = [col['name'] for col in st.session_state.master_columns]
                            for col in master_cols:
                                if col in x.columns:
                                    styles[col] = 'background-color: #e6f2ff'
                        
                        # Age column (light yellow)
                        if st.session_state.age_column:
                            age_col = st.session_state.age_column['name']
                            if age_col in x.columns:
                                styles[age_col] = 'background-color: #ffffcc'
                        
                        # Age group columns (light orange)
                        # Search for both original data columns and generated columns
                        existing_age_group_cols = [col for col in x.columns if 'age group' in col.lower() or 'Age group' in col]
                        generated_age_group_cols = [col for col in x.columns if 'Generated Age Group' in col]
                        all_age_group_cols = existing_age_group_cols + generated_age_group_cols
                        
                        # Apply styling for all age group columns
                        for col in all_age_group_cols:
                            # Use slightly different color for original vs generated columns
                            if col in generated_age_group_cols:
                                styles[col] = 'background-color: #ffd699'  # Darker orange for generated columns
                            else:
                                styles[col] = 'background-color: #ffebcc'  # Light orange for existing columns
                        
                        # Gender column (light pink)
                        if st.session_state.gender_column:
                            gender_col = st.session_state.gender_column['name']
                            if gender_col in x.columns:
                                styles[gender_col] = 'background-color: #ffdddd'
                        
                        # Pair columns (light green shades for pre, light purple for post)
                        if st.session_state.pair_areas:
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
                    
                    # First apply column highlighting
                    styled_df = styled_df.apply(highlight_columns, axis=None)
                    
                    # Display the styled dataframe with precision preserved
                    # We'll format numbers directly in Streamlit's dataframe display
                    # rather than using Styler.applymap which can cause CSS formatting issues
                    
                    # Streamlit's dataframe will preserve the float precision as is
                    # Use a taller height to show more rows (enough for all 90 rows)
                    st.dataframe(styled_df, height=600)
                    
                    # Show legend for colors
                    st.write("**Column Legend:**")
                    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                    with col1:
                        st.markdown('<div style="background-color: #e6f2ff; padding: 5px;">Master Columns</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div style="background-color: #ffffcc; padding: 5px;">Age Column</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div style="background-color: #ffebcc; padding: 5px;">Original Age Group</div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown('<div style="background-color: #ffd699; padding: 5px;">Generated Age Group</div>', unsafe_allow_html=True)
                    with col5:
                        st.markdown('<div style="background-color: #ffdddd; padding: 5px;">Gender Column</div>', unsafe_allow_html=True)
                    with col6:
                        st.markdown('<div style="background-color: rgba(144, 200, 144, 0.3); padding: 5px;">Pre Pairs</div>', unsafe_allow_html=True)
                    with col7:
                        st.markdown('<div style="background-color: rgba(200, 144, 240, 0.3); padding: 5px;">Post Pairs</div>', unsafe_allow_html=True)
                else:
                    st.warning("None of the configured columns were found in the dataset.")
        

        
        # Configuration summary
        if st.session_state.configuration_complete:
            st.write("### Configuration Summary")
            
            # Get the configuration
            config = {
                'master': st.session_state.master_columns,
                'age': {
                    'column': st.session_state.age_column,
                    'type': st.session_state.age_type
                },
                'gender': st.session_state.gender_column,
                'pairs': st.session_state.pair_areas
            }
            
            st.write("**Master Columns:**")
            if config['master']:
                st.write(", ".join([col['name'] for col in config['master']]))
            else:
                st.write("None (Optional)")
            
            st.write("**Age Column:**")
            if config['age']['column']:
                st.write(f"{config['age']['column']['name']} ({config['age']['type']})")
            else:
                st.write("None (Optional)")
                
            st.write("**Gender Column:**")
            if config['gender'] is not None:
                st.write(f"{config['gender']['name']}")
            else:
                st.write("None (Optional)")
            
            st.write("**Pairs:**")
            if config['pairs']:
                for i, pair in enumerate(config['pairs']):
                    if pair['pre'] or pair['post']:
                        st.write(f"Pair {i+1}:")
                        st.write(f"- Pre: {', '.join([col['name'] for col in pair['pre']])}")
                        st.write(f"- Post: {', '.join([col['name'] for col in pair['post']])}")
            else:
                st.write("None (Optional)")
    else:
        # If no data loaded yet
        st.warning("Please upload an Excel file in the first step to get started.")

if __name__ == "__main__":
    display_drag_drop_ui()