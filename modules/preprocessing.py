import pandas as pd
import os
from pathlib import Path


def convert_text_case(text):
    """
    Convert text to title case with specific handling for acronyms and abbreviation expansion.
    
    Args:
        text (str): Input text to convert
        
    Returns:
        str: Converted text with proper casing and expanded abbreviations
    """
    if pd.isna(text) or text == "":
        return text
    
    # Define abbreviation mappings (abbreviation -> full form)
    abbreviation_mappings = {
        "CKD": "Chronic Kidney Disease",
        "IBD": "Inflammatory Bowel Disease"
    }
    
    # First, check if the entire text is an abbreviation that should be expanded
    text_upper = str(text).upper().strip()
    if text_upper in abbreviation_mappings:
        return abbreviation_mappings[text_upper]
    
    # Define acronyms that should remain uppercase (for other cases)
    acronyms = {"CKD", "IBD"}
    
    # Split by spaces and process each word
    words = str(text).split()
    processed_words = []
    
    for word in words:
        # Check if the word is an acronym (case-insensitive)
        if word.upper() in acronyms:
            processed_words.append(word.upper())
        else:
            # Convert to title case (first letter uppercase, rest lowercase)
            processed_words.append(word.capitalize())
    
    return " ".join(processed_words)


def preprocess_dataframe(df, *, columns):
    """
    Apply preprocessing to specified columns of a DataFrame in memory.
    Converts specified columns to proper case.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list of int): Column indices to preprocess (must be keyword argument)
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with modified specified columns
    """
    if df.empty:
        return df
    
    # Validate columns parameter
    if not isinstance(columns, list) or not all(isinstance(col, int) for col in columns):
        raise ValueError("columns must be a list of integers")
    
    # Get column names
    df_columns = df.columns.tolist()
    
    # Check if all specified column indices are valid
    for col_idx in columns:
        if col_idx < 0 or col_idx >= len(df_columns):
            raise ValueError(f"Column index {col_idx} is out of range. DataFrame has {len(df_columns)} columns.")
    
    # Create a copy to avoid modifying the original
    df_processed = df.copy()

    # Define abbreviation map
    abbreviation_map = {
        "Inflammatory Bowel Disease": "IBD",
        "Chronic Kidney Disease": "CKD",
        "Reflux Disorders": "Reflux Disorders", # Ensure consistency
        "Menopausal And Perimenopausal": "Menopausal And Perimenopausal" # Ensure consistency
    }

    # Apply abbreviations to specified columns
    for col_idx in columns:
        col_name = df_columns[col_idx]
        df_processed[col_name] = df_processed[col_name].replace(abbreviation_map)

    # Apply text case conversion to specified columns
    for col_idx in columns:
        col_name = df_columns[col_idx]
        df_processed[col_name] = df_processed[col_name].apply(convert_text_case)

    return df_processed


def preprocess_csv(csv_path, save_file=False, output_path=None):
    """
    Preprocess a CSV file by converting the first and second columns to proper case.
    
    Args:
        csv_path (str): Path to the input CSV file
        save_file (bool): Whether to save the preprocessed file
        output_path (str, optional): Path to save the output file. If None, uses input path with '_preprocessed' suffix
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    if df.empty:
        raise ValueError("CSV file is empty")
    
    # Get column names
    columns = df.columns.tolist()
    
    if len(columns) < 2:
        raise ValueError("CSV file must have at least 2 columns")
    
    # Process first and second columns
    first_col = columns[0]
    second_col = columns[1]
    
    # Apply text case conversion to first and second columns
    df[first_col] = df[first_col].apply(convert_text_case)
    df[second_col] = df[second_col].apply(convert_text_case)
    
    # Save file if requested
    if save_file:
        if output_path is None:
            # Create output path with '_preprocessed' suffix
            input_path = Path(csv_path)
            output_path = input_path.parent / f"{input_path.stem}_preprocessed{input_path.suffix}"
        
        try:
            df.to_csv(output_path, index=False)
            print(f"Preprocessed file saved to: {output_path}")
        except Exception as e:
            raise ValueError(f"Error saving preprocessed file: {e}")
    
    return df


def preprocess_data_directory(data_dir="data", save_files=False, output_dir=None):
    """
    Preprocess all CSV files in the data directory.
    
    Args:
        data_dir (str): Path to the directory containing CSV files
        save_files (bool): Whether to save preprocessed files
        output_dir (str, optional): Directory to save output files. If None, uses data_dir
        
    Returns:
        dict: Dictionary mapping filenames to preprocessed DataFrames
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Get all CSV files in the directory
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {data_dir}")
    
    processed_data = {}
    
    for csv_file in csv_files:
        try:
            # Determine output path if saving
            if save_files:
                if output_dir is None:
                    output_path = csv_file.parent / f"{csv_file.stem}_preprocessed{csv_file.suffix}"
                else:
                    output_dir_path = Path(output_dir)
                    output_dir_path.mkdir(exist_ok=True)
                    output_path = output_dir_path / f"{csv_file.stem}_preprocessed{csv_file.suffix}"
            else:
                output_path = None
            
            # Process the file
            df = preprocess_csv(str(csv_file), save_file=save_files, output_path=output_path)
            processed_data[csv_file.name] = df
            
            print(f"Processed: {csv_file.name}")
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    return processed_data


# Example usage and testing
if __name__ == "__main__":
    # Test the text conversion function
    test_cases = [
        "HEART FAILURE",
        "CHRONIC KIDNEY DISEASE",
        "IBD RELATED CONDITIONS", 
        "CKD STAGE 3",
        "diabetes mellitus",
        "INFLAMMATORY BOWEL DISEASE",
        "IBD",  # Test standalone abbreviation
        "CKD",  # Test standalone abbreviation
        "ibd",  # Test lowercase abbreviation
        "ckd"   # Test lowercase abbreviation
    ]
    
    print("Testing text case conversion:")
    for case in test_cases:
        result = convert_text_case(case)
        print(f"'{case}' -> '{result}'")
    
    # Test preprocessing a single file (if data directory exists)
    data_dir = Path("../data")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            sample_file = csv_files[0]
            print(f"\nTesting preprocessing on: {sample_file.name}")
            try:
                df = preprocess_csv(str(sample_file))
                print(f"Successfully processed {sample_file.name}")
                print(f"Shape: {df.shape}")
                print(f"First few rows of first two columns:")
                print(df.iloc[:5, :2])
            except Exception as e:
                print(f"Error: {e}")