"""
Utility module for the DECODE app.

Contains general-purpose helper functions.
"""
import streamlit as st
import re
from datetime import datetime

def get_readable_filename(filename):
    
    
    # Handle individual datasets
    if filename == 'CPRD_Females_45to64.csv':
        return 'CPRD Females 45 to 64 years'
    elif filename == 'CPRD_Females_65plus.csv':
        return 'CPRD Females 65 years and over'
    elif filename == 'CPRD_Females_below45.csv':
        return 'CPRD Females below 45 years'
    elif filename == 'CPRD_Males_45to64.csv':
        return 'CPRD Males 45 to 64 years'
    elif filename == 'CPRD_Males_65plus.csv':
        return 'CPRD Males 65 years and over'
    elif filename == 'CPRD_Males_below45.csv':
        return 'CPRD Males below 45 years'
    if filename == 'SAIL_Females_45to64.csv':
        return 'SAIL Females 45 to 64 years'
    elif filename == 'SAIL_Females_65plus.csv':
        return 'SAIL Females 65 years and over'
    elif filename == 'SAIL_Females_below45.csv':
        return 'SAIL Females below 45 years'
    elif filename == 'SAIL_Males_45to64.csv':
        return 'SAIL Males 45 to 64 years'
    elif filename == 'SAIL_Males_65plus.csv':
        return 'SAIL Males 65 years and over'
    elif filename == 'SAIL_Males_below45.csv':
        return 'SAIL Males below 45 years'
    else:
        return filename

def parse_iqr(iqr_string):
    """Parse IQR string of format 'median [Q1-Q3]' into (median, q1, q3)"""
    try:
        median_str, iqr = iqr_string.split(' [')
        q1, q3 = iqr.strip(']').split('-')
        return float(median_str), float(q1), float(q3)
    except:
        return 0.0, 0.0, 0.0

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run or password not correct
    if "password_correct" not in st.session_state:
        # Show input for password
        st.text_input(
            "Please enter the password",
            type="password",
            key="password",
            on_change=password_entered
        )
        return False
    
    elif not st.session_state["password_correct"]:
        # Password was incorrect, show error and input again
        st.error("❌ Incorrect password. Please try again.")
        st.text_input(
            "Please enter the password",
            type="password",
            key="password",
            on_change=password_entered
        )
        return False

    # Password correct
    elif st.session_state["password_correct"]:
        return True
        
def add_footer():
    st.markdown(
        """
        <div class="footer">
            <div class="footer-copyright">
                <p>© 2024 DECODE Project. Loughborough University. Funded by the National Institute for Health and Care Research (NIHR). <a href="https://decode-project.org/research/"> DECODE Project Website </a> </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )  

def extract_dataset_info(selected_file):
    """Extract dataset information from filename"""
    filename_lower = selected_file.lower()
    
    # Extract database
    if 'cprd' in filename_lower:
        database = 'CPRD'
    elif 'sail' in filename_lower:
        database = 'SAIL'
    else:
        database = 'Unknown'
    
    # Extract gender
    if 'females' in filename_lower:
        gender = 'Female'
    elif 'males' in filename_lower:
        gender = 'Male'
    else:
        gender = 'Unknown'
    
    # Extract age group (use human-friendly labels)
    if 'below45' in filename_lower:
        age_group = 'below 45'
    elif '45to64' in filename_lower:
        age_group = '45 to 64'
    elif '65plus' in filename_lower:
        age_group = '65 plus'
    else:
        age_group = 'Unknown'
    
    return database, gender, age_group

def sanitize_filename_component(component):
    """Sanitize a component for use in filename"""
    if component is None:
        return ""
    
    # Convert to string and remove/replace problematic characters
    sanitized = str(component)
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)  # Replace forbidden chars with underscore
    sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)     # Replace non-alphanumeric (except dash, underscore, dot)
    sanitized = re.sub(r'_+', '_', sanitized)            # Replace multiple underscores with single
    sanitized = sanitized.strip('_')                      # Remove leading/trailing underscores
    
    return sanitized

def format_conditions_for_filename(conditions, max_length=50):
    """Format condition list for filename with length limit"""
    if not conditions:
        return ""
    
    # Sanitize and abbreviate condition names
    sanitized_conditions = []
    for condition in conditions:
        # Create abbreviations for common long condition names
        condition_abbrevs = {
            'Chronic Kidney Disease': 'CKD',
            'Coronary Heart Disease': 'CHD',
            'Peripheral Vascular Disease': 'PVD',
            'Inflammatory Bowel Disease': 'IBD',
            'Polycystic Ovary Syndrome': 'PCOS',
            'Chronic Airway Diseases': 'CAD',
            'Chronic Arthritis': 'CA',
            'Chronic Pain Conditions': 'CPC',
            'Multiple Sclerosis': 'MS',
            'Mental Illness': 'MI'
        }
        
        abbrev_condition = condition_abbrevs.get(condition, condition)
        sanitized_conditions.append(sanitize_filename_component(abbrev_condition))
    
    conditions_str = '_'.join(sanitized_conditions)
    
    # Truncate if too long
    if len(conditions_str) > max_length:
        conditions_str = conditions_str[:max_length - 3] + '...'
    
    return conditions_str

def generate_export_filename(base_name, selected_file, analysis_params=None):
    """
    Generate informative export filename incorporating dataset and analysis parameters
    
    Args:
        base_name: Base name for the file (e.g., 'sensitivity_analysis', 'condition_combinations')
        selected_file: Currently selected dataset file
        analysis_params: Dictionary of analysis parameters (optional)
            - min_or: Minimum odds ratio
            - min_freq: Minimum frequency 
            - min_percentage: Minimum percentage
            - time_horizon: Time horizon in years
            - time_margin: Time margin
            - selected_conditions: List of selected conditions
            - top_n: Number of top trajectories
            - file_extension: File extension (defaults to 'csv')
    
    Returns:
        Formatted filename string
    """
    if analysis_params is None:
        analysis_params = {}
    
    # Extract dataset information
    database, gender, age_group = extract_dataset_info(selected_file)
    
    # Start building filename components
    filename_parts = [base_name, database, gender, age_group]
    
    # Add analysis-specific parameters
    if 'selected_conditions' in analysis_params and analysis_params['selected_conditions']:
        conditions_str = format_conditions_for_filename(analysis_params['selected_conditions'])
        if conditions_str:
            filename_parts.append(conditions_str)
    
    if 'top_n' in analysis_params and analysis_params['top_n']:
        filename_parts.append(f"top{analysis_params['top_n']}")
    
    if 'min_or' in analysis_params and analysis_params['min_or'] is not None:
        filename_parts.append(f"minOR{analysis_params['min_or']}")
    
    if 'min_freq' in analysis_params and analysis_params['min_freq'] is not None:
        filename_parts.append(f"minfreq{analysis_params['min_freq']}")
    
    if 'min_percentage' in analysis_params and analysis_params['min_percentage'] is not None:
        filename_parts.append(f"minprev{analysis_params['min_percentage']}")
    
    if 'time_horizon' in analysis_params and analysis_params['time_horizon'] is not None:
        filename_parts.append(f"time{analysis_params['time_horizon']}y")
    
    # Get file extension
    file_extension = analysis_params.get('file_extension', 'csv')
    
    # Join all parts with underscores and add extension
    filename = '_'.join(str(part) for part in filename_parts if part)
    
    # Ensure filename isn't too long (most filesystems have 255 char limit)
    max_length = 200  # Leave room for extension
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    return f"{filename}.{file_extension}"
