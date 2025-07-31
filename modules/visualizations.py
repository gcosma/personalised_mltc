"""
Visualization functions for the DECODE app.
"""
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
from modules.config import SYSTEM_COLORS, condition_categories
from modules.utils import parse_iqr
from modules.preprocessing import convert_text_case

def create_sensitivity_plot(results):
    """Create the sensitivity analysis visualization"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x_vals = results['OR_Threshold'].values
    bar_heights = results['Num_Trajectories']

    # Plot bars and lines
    bars = ax1.bar(x_vals, bar_heights, alpha=0.3, color='navy')
    line = ax2.plot(x_vals, results['Coverage_Percent'], 'r-o', linewidth=2)

    # Add scatter plot with variable sizes
    sizes = (results['System_Pairs'] / results['System_Pairs'].max()) * 500
    scatter = ax2.scatter(x_vals, results['Coverage_Percent'], s=sizes, alpha=0.5, color='darkred')

    # Add text annotations
    for i, row in results.iterrows():
        ax1.text(row['OR_Threshold'], bar_heights[i] * 0.5,
                f"Median: {row['Median_Duration']:.1f}y\nIQR: [{row['Q1_Duration']:.1f}-{row['Q3_Duration']:.1f}]",
                ha='center', va='center', fontsize=10)

    # Labels and legend
    ax1.set_xlabel('Minimum Odds Ratio Threshold')
    ax1.set_ylabel('Number of Disease Trajectories')
    ax2.set_ylabel('Population Coverage (%)')

    legend_elements = [
        Patch(facecolor='navy', alpha=0.3, label='Number of Trajectories'),
        Line2D([0], [0], color='r', marker='o', label='Population Coverage %'),
        Line2D([0], [0], marker='o', color='darkred', alpha=0.5,
               label='System Pairs', markersize=10, linestyle='None')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.title('Impact of Odds Ratio Threshold on Disease Trajectory Analysis')
    plt.tight_layout()
    return fig

def create_combinations_plot(results_df):
    """Create the combinations analysis visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))

    top_10 = results_df.nlargest(10, 'Prevalence % (Based on MPF)')
    bars = ax.bar(range(len(top_10)), top_10['Prevalence % (Based on MPF)'])

    # Customize the plot
    ax.set_xticks(range(len(top_10)))
    ax.set_xticklabels(top_10['Combination'], rotation=45, ha='right')
    ax.set_title('Top 10 Condition Combinations by Prevalence')
    ax.set_xlabel('Condition Combinations')
    ax.set_ylabel('Prevalence (%)')

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')

    plt.tight_layout()
    return fig

def create_personalized_analysis(data, patient_conditions, time_horizon=None, time_margin=None, min_or=2.0):
    """Create a personalized analysis of disease trajectories for a patient's conditions"""
    filtered_data = data[data['OddsRatio'] >= min_or].copy()
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    def get_risk_level(odds_ratio):
        if odds_ratio >= 5:
            return "High", "#dc3545"
        elif odds_ratio >= 3:
            return "Moderate", "#ffc107"
        else:
            return "Low", "#28a745"

    html = """
    <style>
        .patient-analysis {
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont;
            margin: 20px 0;
            width: 100%;
            max-width: 100%;
        }
        .condition-section {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #f8f9fa;
            width: 100%;
        }
        .condition-header {
            font-size: 1.2em;
            color: #2c5282;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }
        .trajectory-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            background-color: white;
            font-size: 14px;
        }
        .trajectory-table th {
            background-color: #f5f5f5;
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
            white-space: nowrap;
        }
        .trajectory-table td {
            padding: 10px;
            border: 1px solid #ddd;
            vertical-align: top;
        }
        .risk-badge {
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
            font-weight: bold;
        }
        .system-tag {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: #e2e8f0;
            font-size: 0.9em;
            margin-right: 5px;
        }
        .timeline-indicator {
            font-style: italic;
            color: #666;
        }
        .progression-arrow {
            color: #4a5568;
            font-weight: bold;
        }
        .percentage {
            color: #2d3748;
            font-weight: bold;
        }
        @media (max-width: 1200px) {
            .trajectory-table {
                display: block;
                overflow-x: auto;
                white-space: nowrap;
            }
        }
        .analysis-container {
            max-width: 100%;
            margin: 0 auto;
            padding: 20px;
        }
        .summary-section {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid #e2e8f0;
        }
    </style>
    <div class="patient-analysis">
        <div class="analysis-container">
            <h2>Personalized Disease Trajectory Analysis</h2>
            <div class="summary-section">
                <h3>Current Conditions:</h3>
                <p>""" + ", ".join(f"<span class='system-tag'>{condition_categories.get(cond, 'Other')}</span> {cond}" for cond in patient_conditions) + """</p>
            </div>
    """

    for condition_a in patient_conditions:
        time_filtered_data = filtered_data[
            (filtered_data['ConditionA'] == condition_a) |
            (filtered_data['ConditionB'] == condition_a)
        ]

        if time_horizon and time_margin:
            time_filtered_data = time_filtered_data[
                time_filtered_data['MedianDurationYearsWithIQR'].apply(
                    lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin)
            ]

        if not time_filtered_data.empty:
            system_a = condition_categories.get(condition_a, 'Other')
            html += f"""
            <div class="condition-section">
                <div class="condition-header">
                    <span class="system-tag">{system_a}</span>
                    Progression Paths from {condition_a}
                </div>
                <table class="trajectory-table">
                    <thead>
                        <tr>
                            <th>Risk Level</th>
                            <th>Potential Progression</th>
                            <th>Expected Timeline</th>
                            <th>Statistical Support</th>
                            <th>Progression Details</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for _, row in time_filtered_data.sort_values('OddsRatio', ascending=False).iterrows():
                if row['ConditionA'] == condition_a:
                    other_condition = row['ConditionB']
                    direction_percentage = row['DirectionalPercentage']
                else:
                    other_condition = row['ConditionA']
                    direction_percentage = 100 - row['DirectionalPercentage']

                if other_condition not in patient_conditions:
                    system_b = condition_categories.get(other_condition, 'Other')
                    median, q1, q3 = parse_iqr(row['MedianDurationYearsWithIQR'])
                    prevalence = (row['PairFrequency'] / total_patients) * 100
                    risk_level, color = get_risk_level(row['OddsRatio'])

                    # Parse precedence to determine direction
                    if "precedes" in row['Precedence']:
                        parts = row['Precedence'].split(" precedes ")
                        first_condition = parts[0]
                        second_condition = parts[1]
                        direction = f"{first_condition} <span class='progression-arrow'>→</span> {second_condition}"
                        if first_condition.lower() == row['ConditionA'].lower():
                            percentage = row['DirectionalPercentage']
                        else:
                            percentage = 100 - row['DirectionalPercentage']

                        progression_text = f"""
                            {direction}<br>
                            <span class='percentage'>{percentage:.1f}%</span> of cases follow this pattern
                        """
                    else:
                        direction = f"{condition_a} <span class='progression-arrow'>→</span> {other_condition}"
                        progression_text = f"""
                            {direction}<br>
                            <span class='percentage'>{direction_percentage:.1f}%</span> of cases follow this pattern
                        """

                    html += f"""
                        <tr>
                            <td><span class="risk-badge" style="background-color: {color}">{risk_level}</span></td>
                            <td>
                                <strong>{other_condition}</strong><br>
                                <span class="system-tag">{system_b}</span>
                            </td>
                            <td class="timeline-indicator">
                                Typically {median:.1f} years<br>
                                Range: {q1:.1f} to {q3:.1f} years
                            </td>
                            <td>
                                OR: {row['OddsRatio']:.1f}<br>
                                {row['PairFrequency']} cases ({prevalence:.1f}%)
                            </td>
                            <td>
                                {progression_text}
                            </td>
                        </tr>
                    """

            html += """
                    </tbody>
                </table>
            </div>
            """

    html += """
            <div class="summary-section">
                <h4>Understanding This Analysis:</h4>
                <ul>
                    <li><strong>Risk Level:</strong> Based on odds ratio strength (High: OR≥5, Moderate: OR≥3, Low: OR≥2)</li>
                    <li><strong>Expected Timeline:</strong> Median years and range between which progression typically occurs</li>
                    <li><strong>Statistical Support:</strong> Odds ratio and number of observed cases in the population</li>
                    <li><strong>Progression Details:</strong> Direction of progression and percentage of cases that follow this pattern</li>
                </ul>
            </div>
        </div>
    </div>
    """

    return html

def create_network_visualization(data, min_or, min_freq):
    """Create network visualization with legends with pastel colors matching paper style"""
    net = Network(height="800px", width="100%", bgcolor='white', font_color='black', directed=True)
    
    # Filter data
    filtered_data = data[
        (data['OddsRatio'] >= min_or) &
        (data['PairFrequency'] >= min_freq)
    ].copy()
    # Apply convert_text_case to ConditionA and ConditionB columns
    filtered_data['ConditionA'] = filtered_data['ConditionA'].apply(convert_text_case)
    filtered_data['ConditionB'] = filtered_data['ConditionB'].apply(convert_text_case)

    # Create condition categories legend HTML - Note the color50 for pastel effect
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; 
                border: 1px solid #ccc; font-size: 12px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Condition Categories</div>
    """
    for category, color in sorted(SYSTEM_COLORS.items()):
        if category != "Other":
            legend_html += f"""
            <div style="margin: 2px 0;">
                <div style="display: inline-block; width: 20px; height: 20px; background: {color}50;
                     border: 1px solid {color}; margin-right: 5px;"></div>
                {category}
            </div>
            """
    legend_html += "</div>"

    # Create patient count ranges legend
    count_legend = """
    <div style="position: absolute; top: 10px; right: 10px; background: white; padding: 10px; 
                border: 1px solid #ccc; font-size: 12px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Patient Count Ranges</div>
    """
    
    # Get frequency percentiles for edge widths
    freqs = filtered_data['PairFrequency'].values
    percentiles = np.percentile(freqs, [0, 20, 40, 60, 80, 100])
    
    # Add ranges to legend
    ranges = [
        (percentiles[0], percentiles[1], "0% - 20%"),
        (percentiles[1], percentiles[2], "20% - 40%"),
        (percentiles[2], percentiles[3], "40% - 60%"),
        (percentiles[3], percentiles[4], "60% - 80%"),
        (percentiles[4], percentiles[5], "80%+")
    ]
    
    for i, (lower, upper, label) in enumerate(ranges, 1):
        count_legend += f"""
        <div style="margin: 5px 0;">
            <div style="border-bottom: {i}px solid black; width: 40px; display: inline-block; margin-right: 5px;"></div>
            {int(lower)} ≤ Patients < {int(upper)} ({label})
        </div>
        """
    count_legend += "</div>"

    # Network options for clear visualization
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 14},
            "shape": "dot"
        },
        "edges": {
            "font": {
                "size": 8,
                "align": "middle",
                "background": "white"
            },
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        },
        "physics": {
            "enabled": false,
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
            }
        }
    }
    """)

    # Add nodes with system-based layout and pastel colors
    unique_systems = set(condition_categories[cond] for cond in set(filtered_data['ConditionA']) | set(filtered_data['ConditionB']))
    radius = 300
    system_angles = {sys: i * (2 * math.pi / len(unique_systems)) for i, sys in enumerate(sorted(unique_systems))}
    
    # Add nodes with pastel colors
    for condition in set(filtered_data['ConditionA']) | set(filtered_data['ConditionB']):
        system = condition_categories.get(condition, "Other")
        base_color = SYSTEM_COLORS.get(system, SYSTEM_COLORS["Other"])
        angle = system_angles[system]
        
        # Add random variation to position
        x = radius * math.cos(angle) + random.uniform(-50, 50)
        y = radius * math.sin(angle) + random.uniform(-50, 50)
        
        # Create node with pastel color (using 50% transparency)
        net.add_node(
            condition,
            label=condition,
            title=f"{condition}\nSystem: {system}",
            x=x,
            y=y,
            color={'background': f"{base_color}50", 'border': base_color},
            size=30
        )

    # Add edges
    for _, row in filtered_data.iterrows():
        freq = row['PairFrequency']
        
        # Determine edge width based on frequency percentiles
        if freq < percentiles[1]:
            width = 1
        elif freq < percentiles[2]:
            width = 2
        elif freq < percentiles[3]:
            width = 3
        elif freq < percentiles[4]:
            width = 4
        else:
            width = 5

        # Edge label showing OR and Years
        edge_label = f"OR: {row['OddsRatio']:.1f}\nYears: {row['MedianDurationYearsWithIQR']}"
        
        net.add_edge(
            row['ConditionA'],
            row['ConditionB'],
            label=edge_label,
            title=edge_label,
            width=width,
            arrows={'to': {'enabled': True, 'scaleFactor': 0.5}},
            color={'color': 'rgba(128,128,128,0.7)', 'highlight': 'black'},
            font={'size': 8, 'color': 'black', 'strokeWidth': 2, 'strokeColor': 'white'}
        )

    # Generate final HTML with legends
    html = net.generate_html()
    final_html = html.replace('</body>', f'{legend_html}{count_legend}</body>')
    
    return final_html

def create_patient_count_legend(G):
    """Create a dynamic patient count legend based on actual data values.
    Formats the legend to match the specified style with exact patient ranges and percentages.
    
    Args:
        G: NetworkX graph containing edge data with pair_frequency attribute
        
    Returns:
        str: HTML string containing the formatted legend
    """
    try:
        # Extract pair frequencies from graph edges and convert to integers
        pair_frequencies = [int(data['pair_frequency']) for _, _, data in G.edges(data=True)]
        
        if not pair_frequencies:
            return """<div>No data available for legend</div>"""
        
        # Calculate percentiles and round to integers
        percentiles = np.percentile(pair_frequencies, [0, 20, 40, 60, 80, 100])
        percentiles = [int(round(p)) for p in percentiles]
        
        # Define the CSS styles for the legend
        legend_styles = """
            <style>
                .legend-container {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: white;
                    padding: 15px;
                    border: 1px solid #ccc;
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    z-index: 1000;
                    box-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                }
                .legend-title {
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #333;
                }
                .legend-item {
                    margin: 8px 0;
                    display: flex;
                    align-items: center;
                }
                .legend-line {
                    width: 40px;
                    margin-right: 10px;
                    background-color: black;
                }
                .legend-text {
                    color: #333;
                }
            </style>
        """
        
        # Create the legend container
        legend_html = f"""
            {legend_styles}
            <div class="legend-container">
                <div class="legend-title">Patient Count Ranges</div>
        """
        
        # Add the first four ranges
        for i in range(4):
            line_thickness = i + 1
            legend_html += f"""
                <div class="legend-item">
                    <div class="legend-line" style="height: {line_thickness}px;"></div>
                    <div class="legend-text">
                        {percentiles[i]} ≤ Patients < {percentiles[i+1]} ({i*20}% - {(i+1)*20}%)
                    </div>
                </div>
            """
        
        # Add the final range (80%+)
        legend_html += f"""
                <div class="legend-item">
                    <div class="legend-line" style="height: 5px;"></div>
                    <div class="legend-text">
                        Patients ≥ {percentiles[4]} (80%+)
                    </div>
                </div>
            </div>
        """
        
        return legend_html
        
    except Exception as e:
        print(f"Error creating legend: {str(e)}")
        return """<div>Error creating legend</div>"""


def create_network_graph(data, patient_conditions, min_or, time_horizon=None, time_margin=None):
    """Create network graph matching the personalized analysis visualization with cohort-style edges."""
    # Initialize network with higher resolution settings
    net = Network(height="1200px", width="100%", bgcolor='white', font_color='black', directed=True)

    # Enhanced network options
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 24, "strokeWidth": 2},
            "scaling": {"min": 20, "max": 50}
        },
        "edges": {
            "color": {"inherit": false},
            "font": {
                "size": 18,
                "strokeWidth": 2,
                "align": "middle",
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
            }
        }
    }
    """)

    # Apply initial OR filter
    filtered_data = data[data['OddsRatio'] >= min_or].copy()
    # Apply convert_text_case to ConditionA and ConditionB columns
    filtered_data['ConditionA'] = filtered_data['ConditionA'].apply(convert_text_case)
    filtered_data['ConditionB'] = filtered_data['ConditionB'].apply(convert_text_case)

    total_patients = data['TotalPatientsInGroup'].iloc[0]
    
    # Clean patient_conditions as well
    patient_conditions = [convert_text_case(cond) for cond in patient_conditions]
    
    # Find all connected conditions and their relationships
    connected_conditions = set()
    relationships_to_show = []
    
    for condition_a in patient_conditions:
        condition_relationships = filtered_data[
            (filtered_data['ConditionA'] == condition_a) |
            (filtered_data['ConditionB'] == condition_a)
        ]
        
        if time_horizon is not None and time_margin is not None:
            condition_relationships = condition_relationships[
                condition_relationships['MedianDurationYearsWithIQR'].apply(
                    lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin)
            ]
        
        for _, row in condition_relationships.iterrows():
            other_condition = (row['ConditionB'] if row['ConditionA'] == condition_a 
                             else row['ConditionA'])
            
            if other_condition not in patient_conditions:
                connected_conditions.add(other_condition)
                relationships_to_show.append(row)
    
    active_conditions = set(patient_conditions) | connected_conditions

    # Organize by system
    system_conditions = {}
    for condition in active_conditions:
        category = condition_categories.get(condition, "Other")
        if category not in system_conditions:
            system_conditions[category] = []
        system_conditions[category].append(condition)

    # Calculate positions
    active_categories = {condition_categories[cond] for cond in active_conditions 
                        if cond in condition_categories}
    angle_step = (2 * math.pi) / len(active_categories)
    radius = 500
    system_centers = {}

    for i, category in enumerate(sorted(active_categories)):
        angle = i * angle_step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        system_centers[category] = (x, y)

    # Create body systems legend
    systems_legend_html = """
    <div style="position: absolute; top: 10px; right: 10px; background: white;
                padding: 10px; border: 1px solid #ddd; border-radius: 5px; z-index: 1000;">
        <h3 style="margin-top: 0; margin-bottom: 10px;">Legend</h3>
        <div style="margin-bottom: 10px;">
            <strong>Node Types:</strong><br>
            ★ Initial Condition<br>
            ○ Related Condition
        </div>
        <div>
            <strong>Body Systems:</strong><br>
    """

    for system, color in SYSTEM_COLORS.items():
        systems_legend_html += f"""
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 15px; height: 15px; background-color: {color}50;
                 border: 1px solid {color}; margin-right: 5px;"></div>
            <span>{system}</span>
        </div>
        """

    systems_legend_html += """
        </div>
        <div style="margin-top: 10px;">
            <strong>Edge Information:</strong><br>
            • Edge thickness indicates strength of relationship<br>
            • Arrow indicates typical progression direction<br>
            • Hover over edges for detailed statistics
        </div>
    </div>
    """

    # Add nodes
    for category, conditions in system_conditions.items():
        center_x, center_y = system_centers[category]
        sub_radius = radius / (len(conditions) + 1)
        
        for j, condition in enumerate(conditions):
            sub_angle = (j / len(conditions)) * (2 * math.pi)
            node_x = center_x + sub_radius * math.cos(sub_angle)
            node_y = center_y + sub_radius * math.sin(sub_angle)
            
            is_initial = condition in patient_conditions
            node_label = f"★ {condition}" if is_initial else condition
            node_size = 40 if is_initial else 30
            base_color = SYSTEM_COLORS[category]
            
            net.add_node(
                condition,
                label=node_label,
                title=f"{condition}\nSystem: {category}",
                size=node_size,
                x=node_x,
                y=node_y,
                color={'background': f"{base_color}50", 
                       'border': '#000000' if is_initial else base_color},
                physics=True,
                fixed=False
            )

    # Get frequency percentiles for edge widths
    if relationships_to_show:
        freqs = [row['PairFrequency'] for row in relationships_to_show]
        percentiles = np.percentile(freqs, [0, 20, 40, 60, 80, 100])
    else:
        percentiles = np.zeros(6)  # Default if no relationships

    # Create dynamic edge width legend
    edge_legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background: white;
                padding: 10px; border: 1px solid #ddd; border-radius: 5px; z-index: 1000;">
        <div style="font-weight: bold; margin-bottom: 5px;">Patient Count Ranges</div>
    """
    
    # Add ranges to legend
    ranges = [
        (percentiles[0], percentiles[1], "0% - 20%"),
        (percentiles[1], percentiles[2], "20% - 40%"),
        (percentiles[2], percentiles[3], "40% - 60%"),
        (percentiles[3], percentiles[4], "60% - 80%"),
        (percentiles[4], percentiles[5], "80%+")
    ]
    
    for i, (lower, upper, label) in enumerate(ranges, 1):
        edge_legend_html += f"""
        <div style="margin: 5px 0;">
            <div style="border-bottom: {i}px solid black; width: 40px; 
                       display: inline-block; margin-right: 5px;"></div>
            {int(lower)} ≤ Patients < {int(upper)} ({label})
        </div>
        """
    edge_legend_html += "</div>"

    # Add edges with standardized widths
    processed_edges = set()
    
    for row in relationships_to_show:
        condition_a = row['ConditionA']
        condition_b = row['ConditionB']
        
        edge_pair = tuple(sorted([condition_a, condition_b]))
        if edge_pair in processed_edges:
            continue
        processed_edges.add(edge_pair)

        if "precedes" in row['Precedence']:
            parts = row['Precedence'].split(" precedes ")
            source_from_precedence = convert_text_case(parts[0])
            target_from_precedence = convert_text_case(parts[1])

            # Determine the actual source and target for the edge based on the row's conditions
            # and ensure they are the cleaned versions
            if source_from_precedence == row['ConditionA']:
                source = row['ConditionA']
                target = row['ConditionB']
                percentage = row['DirectionalPercentage']
            else: # source_from_precedence == row['ConditionB']
                source = row['ConditionB']
                target = row['ConditionA']
                percentage = 100 - row['DirectionalPercentage']
        else:
            # If no explicit precedence, assume A -> B and use already cleaned ConditionA/B
            source = row['ConditionA']
            target = row['ConditionB']
            percentage = row['DirectionalPercentage']

        # Calculate edge width based on percentiles
        freq = row['PairFrequency']
        for i, (lower, upper, _) in enumerate(ranges, 1):
            if lower <= freq < upper:
                edge_width = i
                break
        else:
            edge_width = 5  # Maximum width for highest range

        prevalence = (row['PairFrequency'] / total_patients) * 100
        
        edge_label = (
            f"OR: {row['OddsRatio']:.1f}\n"
            f"Years: {row['MedianDurationYearsWithIQR']}\n"
            f"n={row['PairFrequency']} ({prevalence:.1f}%)\n"
            f"Proceeds: {percentage:.1f}%"
        )

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

    # Add export and download functionality
    export_script = """
    <script type="text/javascript">
    function exportHighRes() {
        const network = document.getElementsByTagName('canvas')[0];
        const scale = 3;
        
        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = network.width * scale;
        exportCanvas.height = network.height * scale;
        
        const ctx = exportCanvas.getContext('2d');
        ctx.scale(scale, scale);
        ctx.drawImage(network, 0, 0);
        
        const link = document.createElement('a');
        link.download = 'trajectory_network.png';
        link.href = exportCanvas.toDataURL('image/png');
        link.click();
    }

    function downloadNetwork() {
        const htmlContent = document.documentElement.outerHTML;
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'network.html';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }
    </script>
    """

    buttons_html = """
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;
                display: flex; gap: 10px;">
        <button onclick="exportHighRes()" 
                style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50;
                       color: white; border: none; border-radius: 5px; cursor: pointer;">
            📸 Download High-Res Image
        </button>
        <button onclick="downloadNetwork()" 
                style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50;
                       color: white; border: none; border-radius: 5px; cursor: pointer;">
            📥 Download Network
        </button>
    </div>
    """

    # Generate final HTML with all components
    network_html = net.generate_html()
    final_html = network_html.replace(
        '</head>',
        export_script + '</head>'
    ).replace(
        '</body>',
        f'{systems_legend_html}{edge_legend_html}{buttons_html}</body>'
    )

    return final_html
