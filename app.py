import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import math
import random
import tempfile
import base64
from pathlib import Path

# Complete dictionary of conditions and their categories
condition_categories = {
    "Anaemia": "Blood",
    "Cardiac Arrhythmias": "Circulatory",
    "Coronary Heart Disease": "Circulatory",
    "Heart Failure": "Circulatory",
    "Hypertension": "Circulatory",
    "Peripheral Vascular Disease": "Circulatory",
    "Stroke": "Circulatory",
    "Barretts Oesophagus": "Digestive",
    "Chronic Constipation": "Digestive",
    "Chronic Diarrhoea": "Digestive",
    "Cirrhosis": "Digestive",
    "Dysphagia": "Digestive",
    "Inflammatory Bowel Disease": "Digestive",
    "Reflux Disorders": "Digestive",
    "Hearing Loss": "Ear",
    "Addisons Disease": "Endocrine",
    "Diabetes": "Endocrine",
    "Polycystic Ovary Syndrome": "Endocrine",
    "Thyroid Disorders": "Endocrine",
    "Visual Impairment": "Eye",
    "Chronic Kidney Disease": "Genitourinary",
    "Menopausal and Perimenopausal": "Genitourinary",
    "Dementia": "Mental",
    "Mental Illness": "Mental",
    "Tourette": "Mental",
    "Chronic Arthritis": "Musculoskeletal",
    "Chronic Pain Conditions": "Musculoskeletal",
    "Osteoporosis": "Musculoskeletal",
    "Cancer": "Neoplasms",
    "Cerebral Palsy": "Nervous",
    "Epilepsy": "Nervous",
    "Insomnia": "Nervous",
    "Multiple Sclerosis": "Nervous",
    "Neuropathic Pain": "Nervous",
    "Parkinsons": "Nervous",
    "Bronchiectasis": "Respiratory",
    "Chronic Airway Diseases": "Respiratory",
    "Chronic Pneumonia": "Respiratory",
    "Interstitial Lung Disease": "Respiratory",
    "Psoriasis": "Skin"
}

# System colors
SYSTEM_COLORS = {
    "Blood": "#DC143C",
    "Circulatory": "#FF4500",
    "Digestive": "#32CD32",
    "Ear": "#4169E1",
    "Endocrine": "#BA55D3",
    "Eye": "#20B2AA",
    "Genitourinary": "#DAA520",
    "Mental": "#8B4513",
    "Musculoskeletal": "#4682B4",
    "Neoplasms": "#800080",
    "Nervous": "#FFD700",
    "Respiratory": "#48D1CC",
    "Skin": "#F08080"
}

def parse_iqr(iqr_string):
    """Parse IQR string of format 'median [Q1-Q3]' into (median, q1, q3)"""
    try:
        median_str, iqr = iqr_string.split(' [')
        q1, q3 = iqr.strip(']').split('-')
        return float(median_str), float(q1), float(q3)
    except:
        return 0.0, 0.0, 0.0

def load_and_process_data(uploaded_file):
    """Load and process the uploaded CSV file"""
    try:
        data = pd.read_csv(uploaded_file)
        total_patients = data['TotalPatientsInGroup'].iloc[0]

        # Extract gender and age group from filename
        filename = uploaded_file.name.lower()

        if 'females' in filename:
            gender = 'Female'
        elif 'males' in filename:
            gender = 'Male'
        else:
            gender = 'Unknown Gender'

        if 'below45' in filename:
            age_group = '<45'
        elif '45to64' in filename:
            age_group = '45-64'
        elif '65plus' in filename:
            age_group = '65+'
        else:
            age_group = 'Unknown Age Group'

        return data, total_patients, gender, age_group

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None, None

def perform_sensitivity_analysis(data, or_thresholds=[2.0, 3.0, 4.0, 5.0]):
    """Perform sensitivity analysis on the data"""
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    for threshold in or_thresholds:
        filtered_data = data[data['OddsRatio'] >= threshold].copy()
        n_trajectories = len(filtered_data)

        # Calculate coverage
        total_pairs = filtered_data['PairFrequency'].sum()
        estimated_unique_patients = total_pairs / 2
        coverage = min((estimated_unique_patients / total_patients) * 100, 100.0)

        # Get system pairs
        system_pairs = set()
        for _, row in filtered_data.iterrows():
            sys_a = condition_categories.get(row['ConditionA'], 'Other')
            sys_b = condition_categories.get(row['ConditionB'], 'Other')
            if sys_a != sys_b:
                system_pairs.add(tuple(sorted([sys_a, sys_b])))

        # Calculate duration statistics
        duration_stats = filtered_data['MedianDurationYearsWithIQR'].apply(parse_iqr)
        medians = [x[0] for x in duration_stats if x[0] > 0]
        q1s = [x[1] for x in duration_stats if x[1] > 0]
        q3s = [x[2] for x in duration_stats if x[2] > 0]

        results.append({
            'OR_Threshold': threshold,
            'Num_Trajectories': n_trajectories,
            'Coverage_Percent': round(coverage, 2),
            'System_Pairs': len(system_pairs),
            'Median_Duration': round(np.median(medians) if medians else 0, 2),
            'Q1_Duration': round(np.median(q1s) if q1s else 0, 2),
            'Q3_Duration': round(np.median(q3s) if q3s else 0, 2)
        })

    return pd.DataFrame(results)

def create_network_graph(data, patient_conditions, min_or, time_horizon=None, time_margin=None):
    """Create a network graph visualization"""
    net = Network(notebook=True, bgcolor='white', font_color='black', height="800px")

    # Filter data
    filtered_data = data[data['OddsRatio'] >= min_or].copy()

    # Find connected conditions
    connected_conditions = set()
    for condition_a in patient_conditions:
        time_filtered_data = filtered_data
        if time_horizon and time_margin:
            time_filtered_data = filtered_data[
                (filtered_data['ConditionA'] == condition_a) &
                (filtered_data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin))
            ]
        conditions_b = set(time_filtered_data[time_filtered_data['ConditionA'] == condition_a]['ConditionB'])
        connected_conditions.update(conditions_b)

    # Add nodes
    active_conditions = set(patient_conditions) | connected_conditions
    for condition in active_conditions:
        category = condition_categories.get(condition, "Other")
        color = SYSTEM_COLORS.get(category, "#808080")

        if condition in patient_conditions:
            net.add_node(condition,
                        label=f"★ {condition}",
                        title=f"{condition}\nCategory: {category}",
                        size=30,
                        color={'background': f"{color}50", 'border': '#000000'})
        else:
            net.add_node(condition,
                        label=condition,
                        title=f"{condition}\nCategory: {category}",
                        size=20,
                        color={'background': f"{color}50", 'border': color})

    # Add edges
    total_patients = data['TotalPatientsInGroup'].iloc[0]
    for condition_a in patient_conditions:
        relevant_data = filtered_data[filtered_data['ConditionA'] == condition_a]
        if time_horizon and time_margin:
            relevant_data = relevant_data[
                relevant_data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin)
            ]

        for _, row in relevant_data.iterrows():
            condition_b = row['ConditionB']
            if condition_b not in patient_conditions:
                edge_width = max(1, min(8, math.log2(row['OddsRatio'] + 1)))
                prevalence = (row['PairFrequency'] / total_patients) * 100

                edge_label = (f"OR: {row['OddsRatio']:.1f}\n"
                            f"Years: {row['MedianDurationYearsWithIQR']}\n"
                            f"n={row['PairFrequency']} ({prevalence:.1f}%)")

                net.add_edge(condition_a,
                            condition_b,
                            title=edge_label,
                            width=edge_width,
                            color={'color': 'rgba(128,128,128,0.7)'})

    return net

def main():
    st.set_page_config(page_title="Multimorbidity Analysis Tool", layout="wide")

    st.title("Multimorbidity Analysis Tool")
    st.write("Upload your data file and analyze disease trajectories")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data, total_patients, gender, age_group = load_and_process_data(uploaded_file)

        if data is not None:
            st.sidebar.header("Analysis Parameters")

            # Data summary
            st.sidebar.subheader("Data Summary")
            st.sidebar.write(f"Total patients: {total_patients:,}")
            st.sidebar.write(f"Gender: {gender}")
            st.sidebar.write(f"Age Group: {age_group}")

            # Create tabs for different analyses
            tab1, tab2 = st.tabs(["Sensitivity Analysis", "Trajectory Prediction"])

            with tab1:
                st.header("Sensitivity Analysis")
                if st.button("Run Sensitivity Analysis"):
                    results = perform_sensitivity_analysis(data)

                    # Display results
                    st.dataframe(results)

                    # Create visualization
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    ax2 = ax1.twinx()

                    x_vals = results['OR_Threshold'].values
                    ax1.bar(x_vals, results['Num_Trajectories'], alpha=0.3, color='navy')
                    ax2.plot(x_vals, results['Coverage_Percent'], 'r-o', linewidth=2)

                    ax1.set_xlabel('Minimum Odds Ratio Threshold')
                    ax1.set_ylabel('Number of Disease Trajectories')
                    ax2.set_ylabel('Population Coverage (%)')

                    st.pyplot(fig)

            with tab2:
                st.header("Trajectory Prediction")

                # Parameters
                min_or = st.slider("Minimum Odds Ratio", 1.0, 10.0, 2.0, 0.5)

                # Get unique conditions
                unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
                selected_conditions = st.multiselect("Select Initial Conditions", unique_conditions)

                if selected_conditions:
                    max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]).max())
                    time_horizon = st.slider("Time Horizon (years)", 1, max_years, min(5, max_years))
                    time_margin = st.slider("Time Margin", 0.0, 0.5, 0.1, 0.05)

                    if st.button("Generate Trajectory Network"):
                        net = create_network_graph(data, selected_conditions, min_or, time_horizon, time_margin)

                        # Save and display the network
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                            net.save_graph(tmp.name)
                            with open(tmp.name, 'r', encoding='utf-8') as f:
                                html_content = f.read()

                        st.components.v1.html(html_content, height=800)

                        # Add download button
                        b64 = base64.b64encode(html_content.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="trajectory_network.html">Download Network Graph</a>'
                        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()