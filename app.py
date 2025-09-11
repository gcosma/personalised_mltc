import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from pyvis.network import Network
import math
import random
import tempfile
import base64
from pathlib import Path
from matplotlib import patches
from itertools import combinations
import requests
from io import StringIO
from modules.preprocessing import preprocess_dataframe
from modules.config import INDIVIDUAL_CSV_FILES, condition_categories, SYSTEM_COLORS
from modules.utils import get_readable_filename, parse_iqr, check_password, add_footer
from modules.data_loader import load_and_process_data
from modules.analysis import perform_sensitivity_analysis, analyze_condition_combinations
from modules.visualizations import (
    create_sensitivity_plot, 
    create_combinations_plot, 
    create_personalised_analysis, 
    create_network_visualization, 
    create_patient_count_legend, 
    create_network_graph
)
from modules.ui_tabs import (
    render_sensitivity_tab,
    render_combinations_tab,
    render_personalised_analysis_tab,
    render_trajectory_filter_tab,
    render_cohort_network_tab
)

# Combined dataset options + individual files
CSV_FILES = INDIVIDUAL_CSV_FILES

def clear_session_state():
    """Clear all analysis results from session state when a new file is uploaded"""
    st.session_state.sensitivity_results = None
    st.session_state.network_html = None
    st.session_state.combinations_results = None
    st.session_state.combinations_fig = None
    st.session_state.personalised_html = None
    st.session_state.selected_conditions = []
    st.session_state.min_or = 2.0
    st.session_state.time_horizon = 5
    st.session_state.time_margin = 0.1
    st.session_state.min_frequency = None
    st.session_state.min_percentage = None
    st.session_state.unique_conditions = []
    
    # Clear all slider/textbox widget states to ensure proper reset on dataset change
    widget_keys_to_clear = []
    for key in list(st.session_state.keys()):
        if any(key.endswith(suffix) for suffix in ['_slider', '_input', '_constraint_msg']):
            widget_keys_to_clear.append(key)
    
    for key in widget_keys_to_clear:
        del st.session_state[key]

def main():
    # Initialize session state for data persistence
    if 'sensitivity_results' not in st.session_state:
        st.session_state.sensitivity_results = None
    if 'network_html' not in st.session_state:
        st.session_state.network_html = None
    if 'combinations_results' not in st.session_state:
        st.session_state.combinations_results = None
    if 'combinations_fig' not in st.session_state:
        st.session_state.combinations_fig = None
    if 'personalised_html' not in st.session_state:
        st.session_state.personalised_html = None
    if 'selected_conditions' not in st.session_state:
        st.session_state.selected_conditions = []
    if 'min_or' not in st.session_state:
        st.session_state.min_or = 2.0
    if 'time_horizon' not in st.session_state:
        st.session_state.time_horizon = 5
    if 'time_margin' not in st.session_state:
        st.session_state.time_margin = 0.1
    if 'min_frequency' not in st.session_state:
        st.session_state.min_frequency = None
    if 'min_percentage' not in st.session_state:
        st.session_state.min_percentage = None
    if 'unique_conditions' not in st.session_state:
        st.session_state.unique_conditions = []
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    if 'top_n_trajectories' not in st.session_state:
        st.session_state.top_n_trajectories = 5

    # Page configuration
    st.set_page_config(
        page_title="DECODE-MIDAS: DECODE-MIDAS: Multimorbidity in Intellectual Disability Analysis System",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check password before showing any content
    if not check_password():
        st.stop()

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            width: 100%;
            height: 3rem;
            margin: 1rem 0;
            background-color: #ff4b4b;
            color: white;
        }
        .stButton > button:hover {
            background-color: #ff3333;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            padding: 1rem 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 4rem;
            white-space: pre-wrap;
            background-color: #e2e8f0;
            color: #1a202c;
            border-radius: 4px;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #cbd5e0;
            color: #1a202c;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #2c5282;
            color: white;
        }        
        div[data-testid="stSidebarNav"] {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
        }
        div[data-testid="stFileUploader"] {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 4px;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🏥 DECODE-MIDAS: Multimorbidity in Intellectual Disability Analysis System")
    st.markdown("""
    This tool enables analysis of comorbidity patterns and temporal relationships between multiple long-term conditions in adults with intellectual disability, based on observational data from the United Kingdom.
    """ )

    try:
        # File selection container
        with st.container():
            selected_file = st.selectbox(
                "Select dataset", 
                CSV_FILES,
                format_func=get_readable_filename, 
                help="Choose from available datasets"
            )
            
        if selected_file:
            try:
                # Calculate hash of selected file  
                current_hash = hash(selected_file)
                
                # Check if this is a new file
                if 'data_hash' not in st.session_state or current_hash != st.session_state.data_hash:
                    # Clear all session state variables 
                    clear_session_state()
                    # Update the hash
                    st.session_state.data_hash = current_hash
                
                # Load and process data
                data, total_patients, gender, age_group = load_and_process_data(selected_file)
                
                if data is None:
                    st.error("Failed to load data. Please check your file selection.")
                    st.stop()
                
                # Data summary in an info box
                st.info(f"""
                📊 **Data Summary**  
                - Total Patients: {total_patients:,}
                - Gender: {gender}
                - Age Group: {age_group}
                """)

                # Create tabs with icons
                tabs = st.tabs([
                    "📈 Sensitivity Analysis",
                    "🔍 Condition Combinations",
                    "👤 Personalised Analysis",
                    "🎯 Personalised Trajectory Filter",
                    "🌐 Cohort Network"  
                ])

                with tabs[0]:
                    render_sensitivity_tab(data)
                with tabs[1]:
                    render_combinations_tab(data)
                with tabs[2]:
                    render_personalised_analysis_tab(data)
                with tabs[3]:
                    render_trajectory_filter_tab(data)
                with tabs[4]:
                    render_cohort_network_tab(data)

            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.stop()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()
        
    add_footer()     

if __name__ == "__main__":
    main()
