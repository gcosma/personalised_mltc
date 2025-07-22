#!/usr/bin/env python3
"""
Script to extract individual conditions from SAIL condition pair data.
For each SAIL file, extracts unique conditions with their counts and prevalences.

Run from parent directory: python tools/extract_individual_conditions.py
"""

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict


def extract_individual_conditions(csv_path):
    """
    Extract individual conditions from a CSV file with condition pairs (SAIL or CPRD).
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with columns: Condition, Count, Prevalence
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get total patients (same for all rows, so just take the first value)
    total_patients = df['TotalPatientsInGroup'].iloc[0]
    
    # Dictionary to store condition counts
    condition_counts = defaultdict(int)
    
    # Process each row to extract conditions and their counts
    for _, row in df.iterrows():
        condition_a = row['ConditionA']
        condition_b = row['ConditionB']
        count_a = row['ConditionA_Count']
        count_b = row['ConditionB_Count']
        
        # Add counts for each condition
        # Since we're looking at pairs, a condition might appear multiple times
        # We want the unique count for each condition, so we track the maximum count seen
        condition_counts[condition_a] = max(condition_counts[condition_a], count_a)
        condition_counts[condition_b] = max(condition_counts[condition_b], count_b)
    
    # Create result DataFrame
    conditions = []
    counts = []
    prevalences = []
    
    for condition, count in condition_counts.items():
        conditions.append(condition)
        counts.append(count)
        prevalences.append(count / total_patients)
    
    # Create DataFrame and sort by count (descending)
    result_df = pd.DataFrame({
        'Condition': conditions,
        'Count': counts,
        'Prevalence': prevalences
    })
    
    # Sort alphabetically by condition name
    result_df = result_df.sort_values('Count', ascending=False).reset_index(drop=True)
    
    return result_df


def process_all_files(data_dir="./data/preprocessed", output_dir="./data/individual_conditions"):
    """
    Process all SAIL and CPRD files in the data directory and save individual condition results.
    
    Args:
        data_dir (str): Directory containing CSV files
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get all SAIL and CPRD files
    data_path = Path(data_dir)
    sail_files = list(data_path.glob("SAIL_*.csv"))
    cprd_files = list(data_path.glob("CPRD_*.csv"))
    
    all_files = sail_files + cprd_files
    
    if not all_files:
        print(f"No SAIL or CPRD files found in {data_dir}")
        return
    
    print(f"Found {len(all_files)} files to process:")
    print(f"  - SAIL files: {len(sail_files)}")
    print(f"  - CPRD files: {len(cprd_files)}")
    for file in all_files:
        print(f"  - {file.name}")
    
    # Process each file
    for data_file in all_files:
        try:
            print(f"\nProcessing {data_file.name}...")
            
            # Extract individual conditions
            result_df = extract_individual_conditions(str(data_file))
            
            # Create output filename (cleaner naming)
            output_filename = f"{data_file.stem}.csv"
            output_path = Path(output_dir) / output_filename
            
            # Save results
            result_df.to_csv(output_path, index=False)
            
            print(f"  - Found {len(result_df)} unique conditions")
            print(f"  - Saved to: {output_path}")
            
            # Show top 5 conditions:
            print("  - Top 5 conditions:")
            for i, row in result_df.head(5).iterrows():
                print(f"    {i+1}. {row['Condition']}: {row['Count']} ({row['Prevalence']:.4f})")
            
        except Exception as e:
            print(f"Error processing {data_file.name}: {e}")
            continue
    
    print(f"\nAll files processed. Results saved to: {output_dir}")


def main():
    """Main function to process all files."""
    process_all_files()


if __name__ == "__main__":
    main()