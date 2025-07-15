"""
Core analysis functions for the DECODE app.
"""
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from modules.utils import parse_iqr
from modules.utils import parse_iqr
from modules.config import condition_categories

@st.cache_data
def perform_sensitivity_analysis(data, top_n=5):
    """Perform sensitivity analysis with configurable number of top trajectories"""
    or_thresholds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    # Get top n patterns from full dataset first
    top_patterns = data.nlargest(top_n, 'OddsRatio')[
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

@st.cache_data
def analyze_condition_combinations(data, min_percentage, min_frequency):
    """Analyze combinations of conditions"""
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    filtered_data = data[
        (data['Percentage'] >= min_percentage) &
        (data['PairFrequency'] >= min_frequency)
    ].copy()

    # Clean condition names
    for col in ['ConditionA', 'ConditionB']:
        filtered_data[col] = (filtered_data[col]
                            .str.replace(r'\s*\([^)]*\)', '', regex=True)
                            .str.replace('_', ' '))

    unique_conditions = pd.unique(filtered_data[['ConditionA', 'ConditionB']].values.ravel('K'))

    # Calculate frequencies
    pair_frequency_map = {}
    condition_frequency_map = {}

    for _, row in filtered_data.iterrows():
        for key in [f"{row['ConditionA']}_{row['ConditionB']}",
                   f"{row['ConditionB']}_{row['ConditionA']}"]:
            pair_frequency_map[key] = row['PairFrequency']

        for condition in [row['ConditionA'], row['ConditionB']]:
            condition_frequency_map[condition] = (
                condition_frequency_map.get(condition, 0) + row['PairFrequency']
            )

    # Analyze combinations
    result_data = []
    for k in range(3, min(8, len(unique_conditions) + 1)):
        for comb in combinations(unique_conditions, k):
            pair_frequencies = [
                pair_frequency_map.get(f"{a}_{b}", 0)
                for a, b in combinations(comb, 2)
            ]

            frequency = min(pair_frequencies)
            prevalence = (frequency / total_patients) * 100

            # Calculate odds ratio
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
    results_df = (results_df[results_df['Prevalence of the combination (%)'] > 0]
                 .sort_values('Prevalence of the combination (%)', ascending=False))

    return results_df
