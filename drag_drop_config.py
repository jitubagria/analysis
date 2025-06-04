import streamlit as st
import pandas as pd
import re

def initialize_session_state():
    """Initialize session state variables for the drag-drop configuration"""
    if 'available_columns' not in st.session_state:
        st.session_state.available_columns = []
    if 'master_columns' not in st.session_state:
        st.session_state.master_columns = []
    if 'age_column' not in st.session_state:
        st.session_state.age_column = None
    if 'age_type' not in st.session_state:
        st.session_state.age_type = 'years'
    if 'gender_column' not in st.session_state:
        st.session_state.gender_column = None
    if 'pair_areas' not in st.session_state:
        st.session_state.pair_areas = [{'id': 'pair-1', 'pre': [], 'post': []}]
    if 'configuration_complete' not in st.session_state:
        st.session_state.configuration_complete = False

def process_excel_columns(df):
    """Process uploaded excel file and extract column names"""
    st.session_state.available_columns = [{'id': f'f{i}', 'name': col} for i, col in enumerate(df.columns)]
    st.session_state.master_columns = []
    st.session_state.age_column = None
    st.session_state.gender_column = None
    st.session_state.pair_areas = [{'id': 'pair-1', 'pre': [], 'post': []}]
    st.session_state.configuration_complete = False

def move_column(column, source, destination):
    """Move a column from source to destination"""
    if source == 'available':
        st.session_state.available_columns = [c for c in st.session_state.available_columns if c['id'] != column['id']]
    elif source == 'master':
        st.session_state.master_columns = [c for c in st.session_state.master_columns if c['id'] != column['id']]
    elif source == 'age':
        st.session_state.age_column = None
    elif source == 'gender':
        st.session_state.gender_column = None
    elif source.startswith('pair'):
        match = re.match(r'pair-(\d+)-(pre|post)', source)
        if match:
            pair_idx = int(match.group(1)) - 1
            area_type = match.group(2)
            if 0 <= pair_idx < len(st.session_state.pair_areas):
                st.session_state.pair_areas[pair_idx][area_type] = [
                    c for c in st.session_state.pair_areas[pair_idx][area_type] 
                    if c['id'] != column['id']
                ]
    
    if destination == 'available':
        if not any(c['id'] == column['id'] for c in st.session_state.available_columns):
            st.session_state.available_columns.append(column)
    elif destination == 'master':
        if not any(c['id'] == column['id'] for c in st.session_state.master_columns):
            st.session_state.master_columns.append(column)
    elif destination == 'age':
        st.session_state.age_column = column
    elif destination == 'gender':
        st.session_state.gender_column = column
    elif destination.startswith('pair'):
        match = re.match(r'pair-(\d+)-(pre|post)', destination)
        if match:
            pair_idx = int(match.group(1)) - 1
            area_type = match.group(2)
            if 0 <= pair_idx < len(st.session_state.pair_areas):
                if not any(c['id'] == column['id'] for c in st.session_state.pair_areas[pair_idx][area_type]):
                    st.session_state.pair_areas[pair_idx][area_type].append(column)

def add_pair():
    """Add a new pair area"""
    next_id = len(st.session_state.pair_areas) + 1
    st.session_state.pair_areas.append({
        'id': f'pair-{next_id}', 
        'pre': [], 
        'post': []
    })

def remove_pair(idx):
    """Remove a pair area"""
    if idx < len(st.session_state.pair_areas):
        # Return columns to available area
        for area_type in ['pre', 'post']:
            for column in st.session_state.pair_areas[idx][area_type]:
                if not any(c['id'] == column['id'] for c in st.session_state.available_columns):
                    st.session_state.available_columns.append(column)
        
        # Remove the pair
        st.session_state.pair_areas.pop(idx)

def get_configuration():
    """Get the current configuration as a dictionary"""
    return {
        'master': st.session_state.master_columns,
        'age': {
            'column': st.session_state.age_column,
            'type': st.session_state.age_type
        },
        'gender': st.session_state.gender_column,
        'pairs': st.session_state.pair_areas
    }

def display_config_tool():
    """Display the drag & drop configuration tool"""
    st.title("Data Configuration Tool")
    st.write("Upload your Excel file and set up your analysis configuration.")
    
    # Initialize session state
    initialize_session_state()
    
    # Check if we already have data in the main app's session state
    if 'data' in st.session_state and st.session_state.data is not None:
        df = st.session_state.data
        
        # Process the data if it hasn't been processed yet for this config tool
        if 'file_processed' not in st.session_state or st.session_state.file_processed != st.session_state.filename:
            st.session_state.file_processed = st.session_state.filename
            process_excel_columns(df)
            st.success(f"Using already loaded file: {st.session_state.filename}")
            
        # Show a preview of the data
        st.subheader("Data Preview")
        st.dataframe(df.head())
    else:
        # File Upload backup option if no data in main session state
        st.warning("No data found in session. Please upload an Excel file.")
        
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls', 'csv'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if 'file_processed' not in st.session_state or st.session_state.file_processed != uploaded_file.name:
                    st.session_state.file_processed = uploaded_file.name
                    process_excel_columns(df)
                    st.success(f"Successfully loaded {uploaded_file.name} with {len(df.columns)} columns")
                    
                    # Also update the main app's session state
                    st.session_state.data = df
                    st.session_state.filename = uploaded_file.name
                
                # Show a preview of the data
                st.subheader("Data Preview")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return
        else:
            st.info("Please upload an Excel file to get started.")
            return

    # Available columns area
    with st.expander("Available Columns", expanded=True):
        st.write("Select columns to move them to different areas:")
        if st.session_state.available_columns:
            cols = st.columns(3)
            col_idx = 0
            for column in st.session_state.available_columns:
                with cols[col_idx]:
                    destinations = [
                        "Master Area", 
                        "Age Field", 
                        "Gender Field"
                    ]
                    for i, pair in enumerate(st.session_state.pair_areas):
                        destinations.extend([f"Pair {i+1} Pre", f"Pair {i+1} Post"])
                    
                    destination = st.selectbox(
                        f"Move '{column['name']}'", 
                        ["Keep here"] + destinations,
                        key=f"move_{column['id']}"
                    )
                    
                    if destination != "Keep here":
                        if destination == "Master Area":
                            move_column(column, 'available', 'master')
                        elif destination == "Age Field":
                            move_column(column, 'available', 'age')
                        elif destination == "Gender Field":
                            move_column(column, 'available', 'gender')
                        elif destination.startswith("Pair"):
                            match = re.match(r'Pair (\d+) (Pre|Post)', destination)
                            if match:
                                pair_idx = int(match.group(1))
                                area_type = match.group(2).lower()
                                move_column(column, 'available', f'pair-{pair_idx}-{area_type}')
                        st.rerun()
                
                col_idx = (col_idx + 1) % 3
        else:
            st.info("No available columns. All columns have been assigned.")
    
    # Master Area
    with st.expander("Master Area (Reusable Fields)", expanded=True):
        st.write("These columns can be used across multiple analyses:")
        if st.session_state.master_columns:
            cols = st.columns(4)
            for i, column in enumerate(st.session_state.master_columns):
                with cols[i % 4]:
                    st.button(f"📊 {column['name']}", key=f"master_{column['id']}")
                    if st.button("Return", key=f"return_master_{column['id']}"):
                        move_column(column, 'master', 'available')
                        st.rerun()
        else:
            st.info("No columns assigned to Master Area yet.")
    
    # Age Field
    with st.expander("Age Field", expanded=True):
        st.write("Select a column representing age:")
        if st.session_state.age_column:
            st.write(f"📅 Age Column: **{st.session_state.age_column['name']}**")
            st.radio("Age Type", ["Years", "Months", "Days"], 
                     index=["years", "months", "days"].index(st.session_state.age_type),
                     key="age_type_radio",
                     on_change=lambda: setattr(st.session_state, 'age_type', 
                                            st.session_state.age_type_radio.lower()))
            if st.button("Remove Age Field"):
                move_column(st.session_state.age_column, 'age', 'available')
                st.rerun()
        else:
            st.info("No age column selected yet.")
    
    # Gender Field
    with st.expander("Gender Field", expanded=True):
        st.write("Select a column representing gender:")
        if st.session_state.gender_column:
            st.write(f"👤 Gender Column: **{st.session_state.gender_column['name']}**")
            if st.button("Remove Gender Field"):
                move_column(st.session_state.gender_column, 'gender', 'available')
                st.rerun()
        else:
            st.info("No gender column selected yet.")
    
    # Pairs Area
    st.subheader("Pairs Area (Pre/Post)")
    for i, pair in enumerate(st.session_state.pair_areas):
        with st.expander(f"Pair {i+1}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Pre")
                for column in pair['pre']:
                    st.button(f"⬅️ {column['name']}", key=f"pre_{pair['id']}_{column['id']}")
                    if st.button("Return", key=f"return_pre_{pair['id']}_{column['id']}"):
                        move_column(column, f"{pair['id']}-pre", 'available')
                        st.rerun()
            
            with col2:
                st.write("Post")
                for column in pair['post']:
                    st.button(f"➡️ {column['name']}", key=f"post_{pair['id']}_{column['id']}")
                    if st.button("Return", key=f"return_post_{pair['id']}_{column['id']}"):
                        move_column(column, f"{pair['id']}-post", 'available')
                        st.rerun()
        
        if i > 0 and st.button(f"Remove Pair {i+1}", key=f"remove_pair_{i}"):
            remove_pair(i)
            st.rerun()
    
    if st.button("Add Pair"):
        add_pair()
        st.rerun()
    
    # Save button
    if st.button("Complete Configuration"):
        # All fields are now optional
        st.session_state.configuration_complete = True
        st.success("Configuration completed successfully!")
            
    if st.session_state.configuration_complete:
        st.write("### Configuration Summary")
        config = get_configuration()
        
        st.write("**Master Columns:**")
        if config['master']:
            st.write(", ".join([col['name'] for col in config['master']]))
        else:
            st.write("None")
        
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
        for i, pair in enumerate(config['pairs']):
            if pair['pre'] or pair['post']:
                st.write(f"Pair {i+1}:")
                st.write(f"- Pre: {', '.join([col['name'] for col in pair['pre']])}")
                st.write(f"- Post: {', '.join([col['name'] for col in pair['post']])}")

if __name__ == "__main__":
    display_config_tool()