"""
Utility module for the DECODE app.

Contains general-purpose helper functions.
"""
import streamlit as st

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
