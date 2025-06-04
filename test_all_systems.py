import pandas as pd
import numpy as np
import random

# Load the original Excel file
file_path = 'attached_assets/overall_master.xlsx'
df = pd.read_excel(file_path)

# Create copies of the data and change the System value
amicus_df = df.copy()
comtect_df = df.copy()
comtect_df['System'] = 'Comtect'
# Slightly modify some values to make the data different
comtect_df['Pre Hb (g/dL)'] = comtect_df['Pre Hb (g/dL)'] * random.uniform(0.9, 1.1)
comtect_df['Post Hb (g/dL)'] = comtect_df['Post Hb (g/dL)'] * random.uniform(0.9, 1.1)

optia_df = df.copy()
optia_df['System'] = 'Optia'
# Slightly modify some values to make the data different
optia_df['Pre Hb (g/dL)'] = optia_df['Pre Hb (g/dL)'] * random.uniform(0.9, 1.1)
optia_df['Post Hb (g/dL)'] = optia_df['Post Hb (g/dL)'] * random.uniform(0.9, 1.1)

# Combine all three dataframes
combined_df = pd.concat([amicus_df, comtect_df, optia_df], ignore_index=True)

# Verify the combined dataframe
print(f"Shape of combined data: {combined_df.shape}")
print(f"System values: {combined_df['System'].value_counts().to_dict()}")

# Save the combined dataframe as a new Excel file
output_path = 'attached_assets/all_systems_test.xlsx'
combined_df.to_excel(output_path, index=False)
print(f"Created test file with all three systems at: {output_path}")