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

# Disease Systems Categories
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

# Risk level definitions
RISK_LEVELS = {
    "High": {"threshold": 5.0, "color": "#dc3545"},      # Red
    "Moderate": {"threshold": 3.0, "color": "#ffc107"},  # Yellow
    "Low": {"threshold": 2.0, "color": "#28a745"}        # Green
}

# Custom CSS for Streamlit interface
CUSTOM_CSS = """
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
    background-color: #f0f2f6;
    border-radius: 4px;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #e6e9ef;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #ff4b4b;
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
.risk-badge {
    padding: 4px 8px;
    border-radius: 4px;
    color: white;
    font-weight: bold;
    display: inline-block;
    margin: 2px;
}
.risk-high {
    background-color: #dc3545;
}
.risk-moderate {
    background-color: #ffc107;
}
.risk-low {
    background-color: #28a745;
}
.network-legend {
    position: absolute;
    top: 10px;
    right: 10px;
    background: white;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    z-index: 1000;
}
</style>
"""

# Helper function to parse IQR strings
def parse_iqr(iqr_string):
    """Parse IQR string of format 'median [Q1-Q3]' into (median, q1, q3)"""
    try:
        median_str, iqr = iqr_string.split(' [')
        q1, q3 = iqr.strip(']').split('-')
        return float(median_str), float(q1), float(q3)
    except:
        return 0.0, 0.0, 0.0

# Risk assessment function
def get_risk_level(odds_ratio):
    """Determine risk level and color based on odds ratio thresholds"""
    for level, info in RISK_LEVELS.items():
        if odds_ratio >= info["threshold"]:
            return level, info["color"]
    return "Low", RISK_LEVELS["Low"]["color"]

@st.cache_data
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

@st.cache_data
def perform_sensitivity_analysis(data):
    """Perform sensitivity analysis with risk level calculations"""
    or_thresholds = [2.0, 3.0, 4.0, 5.0]
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    top_patterns = data.nlargest(5, 'OddsRatio')[
        ['ConditionA', 'ConditionB', 'OddsRatio', 'PairFrequency',
         'MedianDurationYearsWithIQR', 'DirectionalPercentage', 'Precedence']
    ].to_dict('records')

    for pattern in top_patterns:
        risk_level, _ = get_risk_level(pattern['OddsRatio'])
        pattern['RiskLevel'] = risk_level

    for threshold in or_thresholds:
        filtered_data = data[data['OddsRatio'] >= threshold].copy()
        n_trajectories = len(filtered_data)

        total_pairs = filtered_data['PairFrequency'].sum()
        estimated_unique_patients = total_pairs / 2
        coverage = min((estimated_unique_patients / total_patients) * 100, 100.0)

        system_pairs = set()
        risk_distribution = {'High': 0, 'Moderate': 0, 'Low': 0}

        for _, row in filtered_data.iterrows():
            sys_a = condition_categories.get(row['ConditionA'], 'Other')
            sys_b = condition_categories.get(row['ConditionB'], 'Other')
            if sys_a != sys_b:
                system_pairs.add(tuple(sorted([sys_a, sys_b])))
            
            risk_level, _ = get_risk_level(row['OddsRatio'])
            risk_distribution[risk_level] += 1

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
            'High_Risk_Count': risk_distribution['High'],
            'Moderate_Risk_Count': risk_distribution['Moderate'],
            'Low_Risk_Count': risk_distribution['Low'],
            'Top_Patterns': top_patterns
        })

    return pd.DataFrame(results)

@st.cache_data
def create_network_graph(data, patient_conditions, min_or, time_horizon=None, time_margin=None):
    """Create network graph with risk level visualization"""
    net = Network(height="800px", width="100%", bgcolor='white', font_color='black', directed=True)
    
    legend_html = create_legend_html()
    
    filtered_data = data[data['OddsRatio'] >= min_or].copy()
    connected_conditions = set()
    condition_risks = {}
    
    for condition_a in patient_conditions:
        time_filtered_data = filtered_data[filtered_data['ConditionA'] == condition_a]
        if time_horizon and time_margin:
            time_filtered_data = time_filtered_data[
                time_filtered_data['MedianDurationYearsWithIQR'].apply(
                    lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin)
            ]
        
        for _, row in time_filtered_data.iterrows():
            condition_b = row['ConditionB']
            risk_level, risk_color = get_risk_level(row['OddsRatio'])
            
            if condition_b not in connected_conditions:
                connected_conditions.add(condition_b)
                condition_risks[condition_b] = {'risk_level': risk_level, 'color': risk_color}
            
            # Add edges with risk information
            edge_width = max(1, min(8, math.log2(row['OddsRatio'] + 1)))
            prevalence = (row['PairFrequency'] / data['TotalPatientsInGroup'].iloc[0]) * 100
            
            edge_title = (f"Risk Level: {risk_level}\n"
                         f"OR: {row['OddsRatio']:.1f}\n"
                         f"Timeline: {row['MedianDurationYearsWithIQR']}\n"
                         f"Cases: {row['PairFrequency']} ({prevalence:.1f}%)\n"
                         f"Direction: {row['DirectionalPercentage']:.1f}%")
            
            net.add_edge(condition_a, condition_b,
                        title=edge_title,
                        width=edge_width,
                        color=risk_color)

    # Add nodes with risk information
    for condition in patient_conditions | connected_conditions:
        system = condition_categories.get(condition, 'Other')
        base_color = SYSTEM_COLORS.get(system, '#808080')
        
        if condition in patient_conditions:
            node_size = 30
            node_label = f"★ {condition}"
        else:
            node_size = 20
            node_label = condition
            
        risk_info = condition_risks.get(condition, {'risk_level': 'Low', 'color': RISK_LEVELS['Low']['color']})
        
        net.add_node(condition,
                    label=node_label,
                    title=f"{condition}\nSystem: {system}\nRisk: {risk_info['risk_level']}",
                    size=node_size,
                    color={'background': f"{base_color}50",
                          'border': risk_info['color']},
                    borderWidth=2)

    # Configure network options
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "springLength": 200,
                "springConstant": 0.08
            },
            "minVelocity": 0.75
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)
    
    html_content = net.generate_html()
    final_html = html_content.replace('</body>', f'{legend_html}</body>')
    
    return final_html

def create_legend_html():
    """Create HTML for the network visualization legend"""
    return """
    <div class="network-legend">
        <h3 style="margin-top: 0;">Legend</h3>
        <div style="margin-bottom: 10px;">
            <strong>Risk Levels:</strong><br>
            <span class="risk-badge risk-high">High Risk (OR ≥ 5)</span><br>
            <span class="risk-badge risk-moderate">Moderate Risk (OR ≥ 3)</span><br>
            <span class="risk-badge risk-low">Low Risk (OR ≥ 2)</span>
        </div>
        <div style="margin-bottom: 10px;">
            <strong>Node Types:</strong><br>
            ★ Initial Condition<br>
            ○ Related Condition
        </div>
        <div style="margin-bottom: 10px;">
            <strong>Edge Colors:</strong><br>
            Colors indicate risk level<br>
            Thickness shows association strength
        </div>
    </div>
    """

@st.cache_data
def analyze_condition_combinations(data, min_percentage, min_frequency):
    """Analyze combinations of conditions with risk assessment"""
    total_patients = data['TotalPatientsInGroup'].iloc[0]
    
    filtered_data = data[
        (data['Percentage'] >= min_percentage) &
        (data['PairFrequency'] >= min_frequency)
    ].copy()
    
    unique_conditions = pd.unique(filtered_data[['ConditionA', 'ConditionB']].values.ravel('K'))
    
    # Calculate frequencies and risks
    pair_frequency_map = {}
    condition_frequency_map = {}
    pair_risk_map = {}
    
    for _, row in filtered_data.iterrows():
        pair_key = f"{row['ConditionA']}_{row['ConditionB']}"
        reverse_key = f"{row['ConditionB']}_{row['ConditionA']}"
        
        pair_frequency_map[pair_key] = row['PairFrequency']
        pair_frequency_map[reverse_key] = row['PairFrequency']
        
        risk_level, _ = get_risk_level(row['OddsRatio'])
        pair_risk_map[pair_key] = risk_level
        pair_risk_map[reverse_key] = risk_level
        
        for condition in [row['ConditionA'], row['ConditionB']]:
            condition_frequency_map[condition] = (
                condition_frequency_map.get(condition, 0) + row['PairFrequency']
            )
    
    result_data = []
    for k in range(3, min(8, len(unique_conditions) + 1)):
        for comb in combinations(unique_conditions, k):
            pair_frequencies = []
            pair_risks = []
            
            for i, cond_a in enumerate(comb):
                for cond_b in comb[i+1:]:
                    pair_key = f"{cond_a}_{cond_b}"
                    if pair_key in pair_frequency_map:
                        pair_frequencies.append(pair_frequency_map[pair_key])
                        pair_risks.append(pair_risk_map[pair_key])
            
            if pair_frequencies:
                frequency = min(pair_frequencies)
                prevalence = (frequency / total_patients) * 100
                
                # Calculate combination risk level
                risk_counts = {'High': 0, 'Moderate': 0, 'Low': 0}
                for risk in pair_risks:
                    risk_counts[risk] += 1
                
                overall_risk = 'High' if risk_counts['High'] > len(pair_risks)/3 else \
                             'Moderate' if risk_counts['Moderate'] > len(pair_risks)/3 else 'Low'
                
                result_data.append({
                    'Combination': ' + '.join(comb),
                    'NumConditions': len(comb),
                    'Minimum Pair Frequency': frequency,
                    'Prevalence (%)': prevalence,
                    'Risk Level': overall_risk,
                    'High Risk Pairs': risk_counts['High'],
                    'Moderate Risk Pairs': risk_counts['Moderate'],
                    'Low Risk Pairs': risk_counts['Low']
                })
    
    results_df = pd.DataFrame(result_data)
    return results_df.sort_values('Prevalence (%)', ascending=False)

def create_sensitivity_plot(results):
    """Create sensitivity analysis visualization with risk distribution"""
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(1, 5)
    
    # Main plot
    ax1 = fig.add_subplot(gs[0, :4])
    ax2 = ax1.twinx()
    
    x_vals = results['OR_Threshold'].values
    
    # Plot bars for trajectory counts
    bars = ax1.bar(x_vals, results['Num_Trajectories'], alpha=0.3, color='navy')
    
    # Plot line for coverage
    line = ax2.plot(x_vals, results['Coverage_Percent'], 'r-o', linewidth=2)
    
    # Add risk distribution
    risk_ax = fig.add_subplot(gs[0, 4])
    for idx, threshold in enumerate(x_vals):
        bottom = 0
        for risk_level, color in [('High', '#dc3545'), ('Moderate', '#ffc107'), ('Low', '#28a745')]:
            count = results[f'{risk_level}_Risk_Count'].iloc[idx]
            risk_ax.bar(threshold, count, bottom=bottom, color=color, alpha=0.7)
            bottom += count
    
    # Customize axes
    ax1.set_xlabel('Minimum Odds Ratio Threshold')
    ax1.set_ylabel('Number of Disease Trajectories')
    ax2.set_ylabel('Population Coverage (%)')
    risk_ax.set_ylabel('Risk Distribution')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='navy', alpha=0.3, label='Trajectories'),
        Line2D([0], [0], color='r', marker='o', label='Coverage %'),
        patches.Patch(facecolor='#dc3545', alpha=0.7, label='High Risk'),
        patches.Patch(facecolor='#ffc107', alpha=0.7, label='Moderate Risk'),
        patches.Patch(facecolor='#28a745', alpha=0.7, label='Low Risk')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(
        page_title="Multimorbidity Analysis Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🏥 Multimorbidity Analysis Tool")
    st.markdown("""
    Analyze disease trajectories and comorbidity patterns with integrated risk assessment.
    Upload your patient data to begin analysis.
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing your patient trajectory data"
    )

    if uploaded_file is not None:
        data, total_patients, gender, age_group = load_and_process_data(uploaded_file)

        if data is not None:
            st.info(f"""
            📊 **Data Summary**
            - Total Patients: {total_patients:,}
            - Gender: {gender}
            - Age Group: {age_group}
            """)

            tabs = st.tabs([
                "📈 Sensitivity Analysis",
                "🔄 Risk-Based Trajectory Analysis",
                "🔍 Condition Combinations"
            ])

            # Sensitivity Analysis Tab
            with tabs[0]:
                st.header("Sensitivity Analysis with Risk Assessment")
                
                analysis_col1, analysis_col2 = st.columns([3, 1])
                
                with analysis_col2:
                    st.markdown("### Analysis Controls")
                    analyze_button = st.button(
                        "🚀 Run Analysis",
                        help="Perform sensitivity analysis with risk assessment"
                    )

                with analysis_col1:
                    if analyze_button:
                        with st.spinner("📊 Analyzing trajectories and risk patterns..."):
                            results = perform_sensitivity_analysis(data)
                            
                            # Display risk distribution summary
                            st.subheader("Risk Distribution Overview")
                            risk_summary = pd.DataFrame({
                                'High Risk': results['High_Risk_Count'],
                                'Moderate Risk': results['Moderate_Risk_Count'],
                                'Low Risk': results['Low_Risk_Count'],
                                'OR Threshold': results['OR_Threshold']
                            }).set_index('OR Threshold')
                            
                            st.dataframe(
                                risk_summary.style.background_gradient(
                                    cmap='RdYlGn_r',
                                    subset=['High Risk', 'Moderate Risk', 'Low Risk']
                                )
                            )

                            # Visualization
                            st.pyplot(create_sensitivity_plot(results))

                            # Top patterns with risk levels
                            st.subheader("Strongest Trajectories by Risk Level")
                            patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                            
                            def risk_color(val):
                                risk_level, color = get_risk_level(val)
                                return f'background-color: {color}; color: white'
                            
                            st.dataframe(
                                patterns_df.style.applymap(
                                    risk_color,
                                    subset=['OddsRatio']
                                )
                            )

            # Trajectory Analysis Tab
            with tabs[1]:
                st.header("Risk-Based Trajectory Analysis")
                
                viz_col, param_col = st.columns([3, 1])
                
                with param_col:
                    st.markdown("### Analysis Parameters")
                    
                    min_or = st.slider(
                        "Minimum Odds Ratio",
                        1.0, 10.0, 2.0, 0.5,
                        help="Filter trajectories by minimum odds ratio"
                    )

                    unique_conditions = sorted(set(data['ConditionA'].unique()) | 
                                            set(data['ConditionB'].unique()))
                    
                    selected_conditions = st.multiselect(
                        "Select Initial Conditions",
                        unique_conditions,
                        help="Choose the starting conditions for analysis"
                    )

                    if selected_conditions:
                        max_years = math.ceil(
                            data['MedianDurationYearsWithIQR'].apply(
                                lambda x: parse_iqr(x)[0]
                            ).max()
                        )
                        
                        time_horizon = st.slider(
                            "Time Horizon (years)",
                            1, max_years, min(5, max_years),
                            help="Maximum time period to consider"
                        )
                        
                        time_margin = st.slider(
                            "Time Margin",
                            0.0, 0.5, 0.1, 0.05,
                            help="Allowable variation in time predictions"
                        )

                        analyze_button = st.button(
                            "🔄 Generate Risk Network",
                            help="Create risk-based trajectory network"
                        )

                with viz_col:
                    if selected_conditions and analyze_button:
                        with st.spinner("🌐 Generating risk-based network..."):
                            try:
                                # Create and display network with risk levels
                                html_content = create_network_graph(
                                    data,
                                    selected_conditions,
                                    min_or,
                                    time_horizon,
                                    time_margin
                                )
                                st.components.v1.html(html_content, height=800)
                                
                                # Add download button
                                st.download_button(
                                    "📥 Download Network Visualization",
                                    html_content,
                                    "trajectory_network.html",
                                    "text/html"
                                )
                            except Exception as e:
                                st.error(f"Error generating network: {str(e)}")

            # Combinations Analysis Tab
            with tabs[2]:
                st.header("Condition Combinations Analysis")
                
                param_col, results_col = st.columns([1, 3])
                
                with param_col:
                    st.markdown("### Analysis Parameters")
                    
                    min_freq = st.slider(
                        "Minimum Pair Frequency",
                        int(data['PairFrequency'].min()),
                        int(data['PairFrequency'].max()),
                        int(data['PairFrequency'].min()),
                        help="Minimum number of occurrences required"
                    )
                    
                    min_prev = st.slider(
                        "Minimum Prevalence (%)",
                        float(data['Percentage'].min()),
                        float(data['Percentage'].max()),
                        float(data['Percentage'].min()),
                        0.1,
                        help="Minimum percentage of population affected"
                    )

                    analyze_button = st.button(
                        "🔍 Analyze Combinations",
                        help="Analyze condition combinations with risk assessment"
                    )

                with results_col:
                    if analyze_button:
                        with st.spinner("🔄 Analyzing condition combinations..."):
                            combinations_df = analyze_condition_combinations(
                                data,
                                min_prev,
                                min_freq
                            )
                            
                            if not combinations_df.empty:
                                st.subheader(f"Found {len(combinations_df)} Combinations")
                                
                                # Style the dataframe with risk colors
                                def risk_background(val):
                                    colors = {
                                        'High': '#dc3545',
                                        'Moderate': '#ffc107',
                                        'Low': '#28a745'
                                    }
                                    return f'background-color: {colors[val]}; color: white'
                                
                                st.dataframe(
                                    combinations_df.style.applymap(
                                        risk_background,
                                        subset=['Risk Level']
                                    )
                                )
                                
                                # Add download button
                                csv = combinations_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Results",
                                    csv,
                                    "combination_analysis.csv",
                                    "text/csv"
                                )
                            else:
                                st.warning("No combinations found matching the criteria.")

if __name__ == "__main__":
    main()
