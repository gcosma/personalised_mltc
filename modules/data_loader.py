"""
Data loading module for the DECODE app.
"""
import streamlit as st
import pandas as pd
import requests
from io import StringIO
from modules.preprocessing import preprocess_dataframe

def load_and_process_data(input_file):
    """Load and process the selected CSV file or combined dataset"""
    try:
        
        
        # Handle individual datasets
        # Use secrets configuration or default to main branch
        branch = st.secrets.get('github_branch', 'main')
        github_url = f"https://raw.githubusercontent.com/gcosma/personalised_mltc/main/data/{input_file}"
        print(f"Attempting to load from URL: {github_url}")  # Debug print
        try:
            response = requests.get(github_url)
            response.raise_for_status()  # Raise an exception for bad status codes  
            print(f"Response status code: {response.status_code}")  # Debug print
            data = pd.read_csv(StringIO(response.text))
            
            # Apply preprocessing to clean up condition names (ConditionA, ConditionB, and Precedence columns)
            data = preprocess_dataframe(data, columns=[0, 1, 13])
        except Exception as e:
            print(f"Error fetching data: {str(e)}")  # Debug print   
            raise

        total_patients = data['TotalPatientsInGroup'].iloc[0]

        # Determine gender and age group from filename  
        filename_lower = input_file.lower()
        
        if 'females' in filename_lower:
            gender = 'Female'
        elif 'males' in filename_lower: 
            gender = 'Male'
        else:
            gender = 'Unknown Gender'

        if 'below45' in filename_lower:
            age_group = '<45' 
        elif '45to64' in filename_lower:
            age_group = '45-64'
        elif '65plus' in filename_lower:
            age_group = '65+'
        else:
            age_group = 'Unknown Age Group'

        return data, total_patients, gender, age_group

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        print(f"Detailed error: {str(e)}")  # Debug print
        import traceback 
        print(traceback.format_exc())  # Print full traceback
        return None, None, None, None
