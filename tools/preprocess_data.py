#!/usr/bin/env python3
"""
Helper script to preprocess all CSV files in ./data/raw and save them to ./data/preprocessed.
Uses enhanced preprocessing from modules/preprocessing.py (including Precedence column).

Run from parent directory: python tools/preprocess_data.py
"""

import os
import sys
from pathlib import Path

# Add the modules directory to the path so we can import preprocessing
# Script should be run from parent directory: python tools/preprocess_data.py
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from preprocessing import preprocess_dataframe
import pandas as pd


def main():
    """Main function to preprocess all CSV files in ./data directory."""
    
    # Define directories (run from parent directory)
    data_dir = "./data/raw"
    output_dir = "./data/preprocessed"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get all CSV files in the raw data directory
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} files to process:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    # Process all CSV files with enhanced preprocessing (columns 0, 1, 13)
    try:
        print(f"\nProcessing CSV files with enhanced preprocessing...")
        processed_count = 0
        
        for csv_file in csv_files:
            print(f"Processing {csv_file.name}...")
            
            # Read the file
            df = pd.read_csv(csv_file)
            
            # Apply enhanced preprocessing (ConditionA, ConditionB, and Precedence columns)
            processed_df = preprocess_dataframe(df, columns=[0, 1, 13])
            
            # Create output filename with _preprocessed suffix
            output_filename = f"{csv_file.stem}_preprocessed{csv_file.suffix}"
            output_path = Path(output_dir) / output_filename
            
            # Save the processed file
            processed_df.to_csv(output_path, index=False)
            print(f"  - Saved to: {output_path}")
            
            processed_count += 1
        
        print(f"\nSuccessfully processed {processed_count} files:")
        print(f"Enhanced preprocessing applied to columns: ConditionA (0), ConditionB (1), Precedence (13)")
        print(f"Preprocessed files saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()