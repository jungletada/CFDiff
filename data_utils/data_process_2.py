import os
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    case_dir = 'case_data2'
    source_folder = f'data/{case_dir}/fluent_data_csv'

    # Initialize dictionaries to store min and max values for each column
    min_values = {}
    max_values = {}

    # Get list of CSV files
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

    # Process first file to initialize min/max dictionaries with column names
    first_file = os.path.join(source_folder, csv_files[0])
    first_data = pd.read_csv(first_file, encoding='utf-8-sig')
    
    for col in first_data.columns:
        min_values[col] = first_data[col].min()
        max_values[col] = first_data[col].max()

    # Process remaining files
    for csv_file in tqdm(csv_files[1:], desc="Processing CSV Files", unit="file"):
        file = os.path.join(source_folder, csv_file)
        data = pd.read_csv(file, encoding='utf-8-sig')
        
        # Update min/max values for each column
        for col in data.columns:
            min_values[col] = min(min_values[col], data[col].min())
            max_values[col] = max(max_values[col], data[col].max())

    # Print results
    print("\nMinimum values for each column:")
    for col, val in min_values.items():
        print(f"{col}: {val}")
        
    print("\nMaximum values for each column:")
    for col, val in max_values.items():
        print(f"{col}: {val}")