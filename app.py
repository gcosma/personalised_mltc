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

# Disease category mappings
condition_categories = {
    "Addisons Disease": "Endocrine",
    "Anaemia": "Blood",
    "Barretts Oesophagus": "Digestive",
    "Bronchiectasis": "Respiratory",
    "Cancer": "Neoplasms",
    "Cardiac Arrhythmias": "Cardiovascular",
    "Cerebral Palsy": "Nervous",
    "Chronic Airway Diseases": "Respiratory",
    "Chronic Arthritis": "Musculoskeletal",
    "Chronic Constipation": "Digestive",
    "Chronic Diarrhoea": "Digestive",
    "Chronic Kidney Disease": "Genitourinary",
    "Chronic Pain Conditions": "Musculoskeletal",
    "Chronic Pneumonia": "Respiratory",
    "Cirrhosis": "Digestive",
    "Coronary Heart Disease": "Cardiovascular",
    "Dementia": "Mental health",
    "Diabetes": "Endocrine",
    "Dysphagia": "Digestive",
    "Epilepsy": "Nervous",
    "Heart Failure": "Cardiovascular",
    "Hearing Loss": "Ear",
    "Hypertension": "Cardiovascular",
    "Inflammatory Bowel Disease": "Digestive",
    "Insomnia": "Nervous",
    "Interstitial Lung Disease": "Respiratory",
    "Mental Illness": "Mental",
    "Menopausal and Perimenopausal": "Genitourinary",
    "Multiple Sclerosis": "Nervous",
    "Neuropathic Pain": "Nervous",
    "Osteoporosis": "Musculoskeletal",
    "Parkinsons": "Nervous",
    "Peripheral Vascular Disease": "Circulatory",
    "Polycystic Ovary Syndrome": "Endocrine",
    "Psoriasis": "Skin",
    "Reflux Disorders": "Digestive",
    "Stroke": "Nervous",
    "Thyroid Disorders": "Endocrine",
    "Tourette": "Mental health",
    "Visual Impairment": "Eye"
}

# System colors for visualization
SYSTEM_COLORS = {
    "Endocrine": "#BA55D3",     # Medium Orchid
    "Blood": "#DC143C",         # Crimson
    "Digestive": "#32CD32",     # Lime Green
    "Respiratory": "#48D1CC",   # Medium Turquoise
    "Neoplasms": "#800080",     # Purple
    "Cardiovascular": "#FF4500", # Orange Red
    "Nervous": "#FFD700",       # Gold
    "Musculoskeletal": "#4682B4", # Steel Blue
    "Genitourinary": "#DAA520", # Goldenrod
    "Mental health": "#8B4513", # Saddle Brown
    "Mental": "#A0522D",       # Sienna
    "Ear": "#4169E1",          # Royal Blue
    "Eye": "#20B2AA",          # Light Sea Green
    "Circulatory": "#FF6347",   # Tomato
    "Skin": "#F08080"          # Light Coral
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

def perform_sensitivity_analysis(data):
    """Perform sensitivity analysis with corrected calculations"""
    or_thresholds = [2.0, 3.0, 4.0, 5.0]
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    # Get top 5 patterns from full dataset first
    top_patterns = data.nlargest(5, 'OddsRatio')[
        ['ConditionA', 'ConditionB', 'OddsRatio', 'PairFrequency',
         'MedianDurationYearsWithIQR', 'DirectionalPercentage', 'Precedence']
    ].to_dict('records')

    for threshold in or_thresholds:
        filtered_data = data[data['OddsRatio'] >= threshold].copy()
        n_trajectories = len(filtered_data)

        total_pairs = filtered_data['PairFrequency'].sum()
        estimated_unique_patients = total_pairs / 2
        coverage = min((estimated_unique_patients / total_patients) * 100, 100.0)

        system_pairs = set()
        for _, row in filtered_data.iterrows():
            sys_a = condition_categories.get(row['ConditionA'], 'Other')
            sys_b = condition_categories.get(row['ConditionB'], 'Other')
            if sys_a != sys_b:
                system_pairs.add(tuple(sorted([sys_a, sys_b])))

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
            'Q3_Duration': round(np.median(q3s) if q3s else 0, 2),
            'Top_Patterns': top_patterns
        })

    return pd.DataFrame(results)

def create_network_graph(data, patient_conditions, min_or, time_horizon=None, time_margin=None):
    """Create network graph for trajectory visualization"""
    net = Network(height="800px", width="100%", bgcolor='white', font_color='black', directed=True)
    
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 16},
            "scaling": {
                "min": 10,
                "max": 30
            }
        },
        "edges": {
            "color": {
                "inherit": false
            },
            "font": {
                "size": 12,
                "align": "middle",
                "multi": true,
                "background": "rgba(255, 255, 255, 0.8)"
            },
            "smooth": {
                "type": "continuous",
                "roundness": 0.2
            }
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -4000,
                "centralGravity": 0.1,
                "springLength": 250,
                "springConstant": 0.03,
                "damping": 0.1,
                "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "stabilization": {
                "enabled": true,
                "iterations": 1000,
                "updateInterval": 25
            },
            "solver": "barnesHut"
        },
        "interaction": {
            "hover": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        }
    }
    """)
    
    filtered_data = data[data['OddsRatio'] >= min_or].copy()
    
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
    
    active_conditions = set(patient_conditions) | connected_conditions
    active_categories = {condition_categories[cond] for cond in active_conditions if cond in condition_categories}

    system_conditions = {}
    for condition in active_conditions:
        category = condition_categories.get(condition, "Other")
        if category not in system_conditions:
            system_conditions[category] = []
        system_conditions[category].append(condition)

    angle_step = (2 * math.pi) / len(active_categories)
    radius = 500
    system_centers = {}

    for i, category in enumerate(sorted(active_categories)):
        angle = i * angle_step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        system_centers[category] = (x, y)

    for category, conditions in system_conditions.items():
        center_x, center_y = system_centers[category]
        sub_radius = radius / (len(conditions) + 1)
        
        for j, condition in enumerate(conditions):
            sub_angle = (j / len(conditions)) * (2 * math.pi)
            node_x = center_x + sub_radius * math.cos(sub_angle)
            node_y = center_y + sub_radius * math.sin(sub_angle)
            
            base_color = SYSTEM_COLORS[category]
            
            if condition in patient_conditions:
                net.add_node(
                    condition,
                    label=f"★ {condition}",
                    title=f"{condition}\nCategory: {category}",
                    size=30,
                    x=node_x,
                    y=node_y,
                    color={'background': f"{base_color}50", 'border': '#000000'},
                    physics=True,
                    fixed=False
                )
            else:
                net.add_node(
                    condition,
                    label=condition,
                    title=f"{condition}\nCategory: {category}",
                    size=20,
                    x=node_x,
                    y=node_y,
                    color={'background': f"{base_color}50", 'border': base_color},
                    physics=True,
                    fixed=False
                )

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
                directional_percentage = row['DirectionalPercentage']

                if directional_percentage >= 50:
                    source, target = condition_a, condition_b
                else:
                    source, target = condition_b, condition_a
                    directional_percentage = 100 - directional_percentage

                edge_label = (f"OR: {row['OddsRatio']:.1f}\n"
                            f"Years: {row['MedianDurationYearsWithIQR']}\n"
                            f"n={row['PairFrequency']} ({prevalence:.1f}%)\n"
                            f"Proceeds: {directional_percentage:.1f}%")

                net.add_edge(
                    source,
                    target,
                    label=edge_label,
                    title=edge_label,
                    width=edge_width,
                    arrows={'to': {'enabled': True, 'scaleFactor': 1}},
                    color={'color': 'rgba(128,128,128,0.7)', 'highlight': 'black'},
                    smooth={'type': 'curvedCW', 'roundness': 0.2}
                )

    return net.generate_html()

def analyze_condition_combinations(data, min_percentage, min_frequency):
    """Analyze combinations of conditions"""
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    filtered_data = data[(data['Percentage'] >= min_percentage) &
                         (data['PairFrequency'] >= min_frequency)].copy()

    filtered_data.loc[:, 'ConditionA'] = filtered_data['ConditionA'].str.replace(r'\s*\([^)]*\)', '', regex=True)
    filtered_data.loc[:, 'ConditionB'] = filtered_data['ConditionB'].str.replace(r'\s*\([^)]*\)', '', regex=True)
    filtered_data.loc[:, 'ConditionA'] = filtered_data['ConditionA'].str.replace('_', ' ')
    filtered_data.loc[:, 'ConditionB'] = filtered_data['ConditionB'].str.replace('_', ' ')

    unique_conditions = pd.unique(filtered_data[['ConditionA', 'ConditionB']].values.ravel('K'))

    pair_frequency_map = {}
    condition_frequency_map = {}

    for _, row in filtered_data.iterrows():
        key1 = f"{row['ConditionA']}_{row['ConditionB']}"
        key2 = f"{row['ConditionB']}_{row['ConditionA']}"
        pair_frequency_map[key1] = row['PairFrequency']
        pair_frequency_map[key2] = row['PairFrequency']

        condition_frequency_map[row['ConditionA']] = condition_frequency_map.get(row['ConditionA'], 0) + row['PairFrequency']
        condition_frequency_map[row['ConditionB']] = condition_frequency_map.get(row['ConditionB'], 0) + row['PairFrequency']

    result_data = []

    for k in range(3, min(8, len(unique_conditions) + 1)):
        for comb in combinations(unique_conditions, k):
            pair_frequencies = [pair_frequency_map.get(f"{a}_{b}", 0) for a, b in combinations(comb, 2)]
            frequency = min(pair_frequencies)
            prevalence = (frequency / total_patients) * 100

            observed = frequency
            expected = total_patients
            for condition in comb:
                expected *= (condition_frequency_map[condition] / total_patients)
            odds_ratio = observed / expected if expected != 0 else float('inf')

            result_data.append({
                'Combination': ' + '.join(comb),
                'NumConditions': len(comb),
                'Minimum Pair Frequency': frequency,
                'Prevalence of the combination (%)': prevalence,
                'Total odds ratio': odds_ratio
            })

    results_df = pd.DataFrame(result_data)
    results_df = results_df.sort_values('Prevalence of the combination (%)', ascending=False)
    results_df = results_df[results_df['Prevalence of the combination (%)'] > 0]

    return results_df

def main():
    st.set_page_config(page_title="Multimorbidity Analysis Tool", layout="wide")

    st.title("Multimorbidity Analysis Tool")
    st.write("Upload your data file and analyze disease trajectories")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data, total_patients, gender, age_group = load_and_process_data(uploaded_file)

        if data is not None:
            # Data summary in main area
            st.write(f"**Data Summary:** {total_patients:,} patients | Gender: {gender} | Age Group: {age_group}")

            tab1, tab2, tab3 = st.tabs(["Sensitivity Analysis", "Trajectory Prediction", "Condition Combinations"])

            with tab1:
                st.header("Sensitivity Analysis")
                st.markdown("""
                This analysis explores how different odds ratio thresholds affect the number of disease trajectories 
                and population coverage in the dataset.
                """)
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Run Sensitivity Analysis", key="run_sensitivity"):
                        with col1:
                            with st.spinner("Performing sensitivity analysis..."):
                                results = perform_sensitivity_analysis(data)

                                st.subheader("Analysis Results")
                                display_df = results.drop('Top_Patterns', axis=1)
                                st.dataframe(display_df)

                                st.subheader("Top 5 Strongest Trajectories")
                                patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                                st.dataframe(patterns_df)

                                fig, ax1 = plt.subplots(figsize=(12, 6))
                                ax2 = ax1.twinx()

                                x_vals = results['OR_Threshold'].values
                                bar_heights = results['Num_Trajectories']

                                bars = ax1.bar(x_vals, bar_heights, alpha=0.3, color='navy')
                                line = ax2.plot(x_vals, results['Coverage_Percent'], 'r-o', linewidth=2)

                                sizes = (results['System_Pairs'] / results['System_Pairs'].max()) * 500
                                scatter = ax2.scatter(x_vals, results['Coverage_Percent'], s=sizes, alpha=0.5, color='darkred')

                                for i, row in results.iterrows():
                                    ax1.text(row['OR_Threshold'], bar_heights[i] * 0.5,
                                            f"Median: {row['Median_Duration']:.1f}y\nIQR: [{row['Q1_Duration']:.1f}-{row['Q3_Duration']:.1f}]",
                                            ha='center', va='center', fontsize=10)

                                ax1.set_xlabel('Minimum Odds Ratio Threshold')
                                ax1.set_ylabel('Number of Disease Trajectories')
                                ax2.set_ylabel('Population Coverage (%)')

                                legend_elements = [
                                    patches.Patch(facecolor='navy', alpha=0.3,
                                                label='Number of Disease Trajectories\n(Height of bars)'),
                                    Line2D([0], [0], color='r', marker='o',
                                           label='Population Coverage %\n(Red line)'),
                                    Line2D([0], [0], marker='o', color='darkred', alpha=0.5,
                                           label='Body System Pairs\n(Size of circles)',
                                           markersize=10, linestyle='None')
                                ]
                                ax1.legend(handles=legend_elements, loc='upper right')

                                plt.title(f'Impact of Odds Ratio Threshold on Disease Trajectory Analysis')
                                plt.tight_layout()
                                st.pyplot(fig)

                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Analysis Results",
                                    data=csv,
                                    file_name="sensitivity_analysis_results.csv",
                                    mime="text/csv"
                                )
            
            with tab2:
                st.header("Trajectory Prediction")
                
                # Split the tab into two columns for parameters and visualization
                viz_col, param_col = st.columns([3, 1])
                
                with param_col:
                    st.subheader("Parameters")
                    min_or = st.slider("Minimum Odds Ratio", 1.0, 10.0, 2.0, 0.5)
                    unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
                    selected_conditions = st.multiselect("Select Initial Conditions", unique_conditions)

                    if selected_conditions:
                        max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]).max())
                        time_horizon = st.slider("Time Horizon (years)", 1, max_years, min(5, max_years))
                        time_margin = st.slider("Time Margin", 0.0, 0.5, 0.1, 0.05)

                        if st.button("Generate Trajectory Network"):
                            with st.spinner("Generating network visualization..."):
                                try:
                                    html_content = create_network_graph(
                                        data, 
                                        selected_conditions, 
                                        min_or, 
                                        time_horizon, 
                                        time_margin
                                    )
                                    
                                    with viz_col:
                                        st.components.v1.html(html_content, height=800)
                                    
                                    st.download_button(
                                        label="Download Network Graph",
                                        data=html_content,
                                        file_name="trajectory_network.html",
                                        mime="text/html"
                                    )
                                except Exception as e:
                                    with viz_col:
                                        st.error(f"Failed to generate network graph: {e}")
            
            with tab3:
                st.header("Condition Combinations Analysis")
                
                # Split into two columns for parameters and results
                param_col, results_col = st.columns([1, 3])
                
                with param_col:
                    st.subheader("Analysis Parameters")
                    
                    min_freq_range = (data['PairFrequency'].min(), data['PairFrequency'].max())
                    min_percentage_range = (data['Percentage'].min(), data['Percentage'].max())

                    min_frequency = st.slider(
                        "Minimum Pair Frequency", 
                        min_value=int(min_freq_range[0]), 
                        max_value=int(min_freq_range[1]), 
                        value=int(min_freq_range[0])
                    )
                    
                    min_percentage = st.slider(
                        "Minimum Prevalence Percentage (%)", 
                        min_value=float(min_percentage_range[0]), 
                        max_value=float(min_percentage_range[1]), 
                        value=float(min_percentage_range[0]),
                        step=0.1
                    )

                    analyze_button = st.button("Analyze Combinations")

                with results_col:
                    if analyze_button:
                        with st.spinner("Analyzing condition combinations..."):
                            results_df = analyze_condition_combinations(data, min_percentage, min_frequency)
                            
                            st.subheader(f"Analysis Results (Total Combinations: {len(results_df)})")
                            st.dataframe(results_df)

                            if not results_df.empty:
                                fig, ax = plt.subplots(figsize=(12, 6))
                                top_10 = results_df.nlargest(10, 'Prevalence of the combination (%)')
                                ax.bar(top_10['Combination'], top_10['Prevalence of the combination (%)'])
                                ax.set_title('Top 10 Condition Combinations by Prevalence')
                                ax.set_xlabel('Condition Combination')
                                ax.set_ylabel('Prevalence (%)')
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                                st.pyplot(fig)

                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Full Results",
                                    data=csv,
                                    file_name="condition_combinations.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No combinations found matching the specified criteria.")

if __name__ == "__main__":
    main()
