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
    "Blood": "#DC143C",         # Crimson
    "Circulatory": "#FF4500",   # Orange Red
    "Digestive": "#32CD32",     # Lime Green
    "Ear": "#4169E1",          # Royal Blue
    "Endocrine": "#BA55D3",     # Medium Orchid
    "Eye": "#20B2AA",          # Light Sea Green
    "Genitourinary": "#DAA520", # Goldenrod
    "Mental": "#8B4513",        # Saddle Brown
    "Musculoskeletal": "#4682B4", # Steel Blue
    "Neoplasms": "#800080",     # Purple
    "Nervous": "#FFD700",       # Gold
    "Respiratory": "#48D1CC",   # Medium Turquoise
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

def perform_sensitivity_analysis(data, or_thresholds=[2.0, 3.0, 4.0, 5.0]):
    """Perform sensitivity analysis on the data"""
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]

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
                
        # Get top 5 patterns
        top_patterns = filtered_data.nlargest(5, 'OddsRatio')[
            ['ConditionA', 'ConditionB', 'OddsRatio', 'PairFrequency',
             'MedianDurationYearsWithIQR', 'DirectionalPercentage', 'Precedence']
        ].to_dict('records')
        
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
    """Create a network graph visualization"""
    net = Network(height="800px", width="100%", bgcolor='white', font_color='black', directed=True)
    
    # Update the network options to match Colab exactly
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 16}
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
            "enabled": false,
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "maxVelocity": 50,
            "solver": "barnesHut"
        },
        "interaction": {
            "hover": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        },
        "manipulation": {
            "enabled": true
        }
    }
    """)
    
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
    
    # Define active conditions and categories
    active_conditions = set(patient_conditions) | connected_conditions
    active_categories = {condition_categories[cond] for cond in active_conditions if cond in condition_categories}

    # Add legend nodes on the left side
    legend_start_x = -600
    legend_start_y = -300
    
    # Add legend title
    net.add_node(
        "legend_title",
        label="Legend",
        x=legend_start_x,
        y=legend_start_y - 50,
        size=0,
        font={'size': 20, 'bold': True},
        physics=False,
        fixed=False
    )
    
    # Add legend items
    for i, category in enumerate(sorted(active_categories)):
        color = SYSTEM_COLORS[category]
        net.add_node(
            f"legend_{category}",
            label=f"{category}",
            x=legend_start_x + 50,
            y=legend_start_y + (i * 60),
            size=10,
            shape='dot',
            color={'background': color, 'border': color},
            font={'size': 16, 'align': 'left'},
            physics=False,
            fixed=True
        )

    # Add condition nodes
    for condition in active_conditions:
        category = condition_categories.get(condition, "Other")
        base_color = SYSTEM_COLORS[category]
        
        if condition in patient_conditions:
            net.add_node(
                condition,
                label=f"★ {condition}",
                title=f"{condition}\nCategory: {category}",
                size=30,
                color={'background': f"{base_color}50", 'border': '#000000'},
                physics=True
            )
        else:
            net.add_node(
                condition,
                label=condition,
                title=f"{condition}\nCategory: {category}",
                size=20,
                color={'background': f"{base_color}50", 'border': base_color},
                physics=True
            )
    
    # Add edges with all information visible
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
    
    # Save to HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        return f.name

def main():
    st.set_page_config(page_title="Multimorbidity Analysis Tool", layout="wide")
    
    st.title("Multimorbidity Analysis Tool")
    st.write("Upload your data file and analyze disease trajectories")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data, total_patients, gender, age_group = load_and_process_data(uploaded_file)
        
        if data is not None:
            st.sidebar.header("Analysis Parameters")
            
            st.sidebar.subheader("Data Summary")
            st.sidebar.write(f"Total patients: {total_patients:,}")
            st.sidebar.write(f"Gender: {gender}")
            st.sidebar.write(f"Age Group: {age_group}")
            
            tab1, tab2 = st.tabs(["Sensitivity Analysis", "Trajectory Prediction"])
            
            with tab1:
                st.header("Sensitivity Analysis")
                if st.button("Run Sensitivity Analysis"):
                    results = perform_sensitivity_analysis(data)
                    
                    # Display results table
                    st.dataframe(results.drop('Top_Patterns', axis=1))
                    
                    # Display top patterns
                    st.subheader("Top 5 Strongest Trajectories")
                    patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                    st.dataframe(patterns_df)
                    
                    # Create visualization
                    # Create visualization
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    ax2 = ax1.twinx()
                    
                    x_vals = results['OR_Threshold'].values
                    bar_heights = results['Num_Trajectories']
                    
                    # Add bars for trajectories
                    bars = ax1.bar(x_vals, bar_heights, alpha=0.3, color='navy')
                    # Add line for coverage
                    line = ax2.plot(x_vals, results['Coverage_Percent'], 'r-o', linewidth=2)
                    
                    # Add scatter for system pairs
                    sizes = (results['System_Pairs'] / results['System_Pairs'].max()) * 500
                    scatter = ax2.scatter(x_vals, results['Coverage_Percent'], s=sizes, alpha=0.5, color='darkred')
                    
                    # Add IQR information inside bars
                    for i, row in results.iterrows():
                        ax1.text(row['OR_Threshold'], bar_heights[i] * 0.5,
                                f"Median: {row['Median_Duration']:.1f}y\nIQR: [{row['Q1_Duration']:.1f}-{row['Q3_Duration']:.1f}]",
                                ha='center', va='center', fontsize=10)
                    
                    ax1.set_xlabel('Minimum Odds Ratio Threshold')
                    ax1.set_ylabel('Number of Disease Trajectories')
                    ax2.set_ylabel('Population Coverage (%)')
                    
                    # Add legend
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
                    
                    if gender and age_group:
                        plt.title(f'Impact of Odds Ratio Threshold on Disease Trajectory Analysis in {gender}s {age_group}')
                    else:
                        plt.title('Impact of Odds Ratio Threshold on Disease Trajectory Analysis in General Population')
                    
                    st.pyplot(fig)
            
            with tab2:
                st.header("Trajectory Prediction")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    min_or = st.slider("Minimum Odds Ratio", 1.0, 10.0, 2.0, 0.5)
                    unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
                    selected_conditions = st.multiselect("Select Initial Conditions", unique_conditions)
                
                if selected_conditions:
                    with col2:
                        max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]).max())
                        time_horizon = st.slider("Time Horizon (years)", 1, max_years, min(5, max_years))
                        time_margin = st.slider("Time Margin", 0.0, 0.5, 0.1, 0.05)
                    
                    if st.button("Generate Trajectory Network"):
                        with st.spinner("Generating network visualization..."):
                            html_file = create_network_graph(data, selected_conditions, min_or, time_horizon, time_margin)
                            
                            # Display legend/instructions
                            st.markdown("""
                            ### How to Read the Graph:
                            - Nodes represent conditions, colored by body system category
                            - ★ marks initial conditions
                            - Edge labels show:
                                - OR: Odds Ratio
                                - Years: Median time [IQR]
                                - n: Patient pairs
                                - Proceeds: Percentage first condition precedes second
                            - Edge thickness represents odds ratio strength
                            - Drag nodes freely to rearrange the network
                            - Use mouse wheel to zoom in/out
                            - Click and drag the background to pan
                            """)
                            
                            # Display network
                            with open(html_file, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            
                            st.components.v1.html(html_content, height=800)
                            
                            # Add download button
                            st.download_button(
                                label="Download Network Graph",
                                data=html_content,
                                file_name="trajectory_network.html",
                                mime="text/html"
                            )

if __name__ == "__main__":
    main()
