import os
import shutil
import tarfile

# Create a backup directory
backup_dir = 'project_backup'
if os.path.exists(backup_dir):
    shutil.rmtree(backup_dir)
os.makedirs(backup_dir)

# Files to backup
important_files = [
    'app.py',
    'ml_utils.py',
    'data_utils.py',
    'analyze_excel.py',
    'stats_utils.py',
    'visualization_utils.py',
    'report_generator.py',
    'drag_drop_config.py',
    'drag_drop_ui.py',
    'excel_analyzer.py'
]

# Copy all files
for filename in important_files:
    if os.path.exists(filename):
        print(f"Backing up {filename}...")
        shutil.copy2(filename, os.path.join(backup_dir, filename))
    else:
        print(f"File {filename} not found, skipping.")

# Create a tar.gz archive
with tarfile.open('project_backup.tar.gz', 'w:gz') as tar:
    tar.add(backup_dir)

print("\nBackup complete! You can now download 'project_backup.tar.gz'")
print("This file contains all your important Python code files.")