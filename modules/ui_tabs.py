"""
UI rendering functions for each tab in the DECODE app.
"""
import streamlit as st
import pandas as pd
import math
from modules.analysis import perform_sensitivity_analysis, analyze_condition_combinations
from modules.visualizations import (
    create_sensitivity_plot, 
    create_combinations_plot, 
    create_personalized_analysis, 
    create_network_visualization, 
    create_network_graph
)
from modules.utils import parse_iqr

def create_slider_with_input(label, min_val, max_val, current_val, step, key_prefix, help_text="", is_float=True, show_tip=False, on_change_callback=None):
    """
    Create a synchronized slider + number input combination.
    
    Args:
        label: Label for the slider
        min_val, max_val: Min/max values
        current_val: Current value
        step: Step size
        key_prefix: Unique prefix for session state keys
        help_text: Help text for both widgets
        is_float: Whether values are float (True) or int (False)
        show_tip: Whether to show the "press enter" tip
    
    Returns:
        The current value from the widgets
    """
    slider_key = f"{key_prefix}_slider"
    input_key = f"{key_prefix}_input"
    
    # Initialize session state if not exists, clamping the initial value
    if slider_key not in st.session_state:
        st.session_state[slider_key] = max(min_val, min(max_val, current_val))
    if input_key not in st.session_state:
        st.session_state[input_key] = max(min_val, min(max_val, current_val))
    
    # Define callback functions
    def on_slider_change():
        # Ensure the input value is clamped to the slider's range
        st.session_state[input_key] = max(min_val, min(max_val, st.session_state[slider_key]))
        if on_change_callback:
            on_change_callback()
    
    def on_input_change():
        # Ensure the slider value is clamped to the input's range
        st.session_state[slider_key] = max(min_val, min(max_val, st.session_state[input_key]))
        if on_change_callback:
            on_change_callback()
    
    # Create columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Remove the value parameter - let Streamlit use session state automatically
        slider_val = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=slider_key,
            on_change=on_slider_change,
            help=help_text
        )
    
    with col2:
        # Also remove value parameter for number_input
        input_val = st.number_input(
            "Or type:",
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=input_key,
            on_change=on_input_change,
            help=f"Press Enter to register. {help_text}" if help_text else "Press Enter to register"
        )
        if show_tip:
            st.caption("💡 Press Enter after typing")
    
    return input_val

def render_sensitivity_tab(data):
    st.header("Sensitivity Analysis")
    st.markdown("""
    Explore how different odds ratio thresholds affect the number of disease
    trajectories and population coverage.
    """)

    main_col, control_col = st.columns([3, 1])

    with control_col:
        with st.container():
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            st.markdown("### Control Panel")
            
            top_n = create_constrained_slider_with_input(
                "Number of Top Trajectories",
                1, 20, 
                st.session_state.top_n_trajectories,
                1, "sensitivity_top_n",
                "Select how many top trajectories to display",
                is_float=False, show_tip=True,
                constraint_max=None,
                constraint_message=""
            )
            st.session_state.top_n_trajectories = int(top_n)
            
            analyse_button = st.button(
                "🚀 Run Analysis",
                key="run_sensitivity",
                help="Click to perform sensitivity analysis"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with main_col:
        try:
            if analyse_button:
                with st.spinner("💫 Analysing data..."):
                    # Clear previous results
                    st.session_state.sensitivity_results = None
                    
                    # Generate new results with selected top_n
                    results = perform_sensitivity_analysis(data, top_n=top_n)
                    st.session_state.sensitivity_results = results

                    display_df = results.drop('Top_Patterns', axis=1)
                    st.subheader("Analysis Results")
                    st.dataframe(
                        display_df.style.background_gradient(cmap='YlOrRd', subset=['Coverage_Percent'])
                    )

                    st.subheader(f"Top {top_n} Strongest Trajectories")
                    patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                    st.dataframe(
                        patterns_df.style.background_gradient(cmap='YlOrRd', subset=['OddsRatio'])
                    )

                    fig = create_sensitivity_plot(results)
                    st.plotly_chart(fig, use_container_width=True)

                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="sensitivity_analysis_results.csv",
                        mime="text/csv"
                    )
            
            # Display existing results if available
            elif st.session_state.sensitivity_results is not None:
                results = st.session_state.sensitivity_results
                display_df = results.drop('Top_Patterns', axis=1)
                
                st.subheader("Analysis Results")
                st.dataframe(
                    display_df.style.background_gradient(cmap='YlOrRd', subset=['Coverage_Percent'])
                )

                num_patterns = len(results.iloc[0]['Top_Patterns'])
                st.subheader(f"Top {num_patterns} Strongest Trajectories")
                patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                st.dataframe(
                    patterns_df.style.background_gradient(cmap='YlOrRd', subset=['OddsRatio'])
                )

                fig = create_sensitivity_plot(results)
                st.plotly_chart(fig, use_container_width=True)

                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="sensitivity_analysis_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error in sensitivity analysis: {str(e)}")
            st.session_state.sensitivity_results = None

def render_combinations_tab(data):
    st.header("Condition Combinations Analysis")
    
    main_col, control_col = st.columns([3, 1])

    with control_col:
        with st.container():
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            st.markdown("### Control Panel")
            
            try:
                min_freq_range = (data['PairFrequency'].min(), data['PairFrequency'].max())
                
                # Initialize session state
                if st.session_state.min_frequency is None:
                    st.session_state.min_frequency = int(min_freq_range[0])
                
                min_frequency = create_constrained_slider_with_input(
                    "Minimum Pair Frequency",
                    int(min_freq_range[0]), int(min_freq_range[1]),
                    st.session_state.min_frequency,
                    1, "combinations_freq",
                    "Minimum number of occurrences required",
                    is_float=False, show_tip=True,
                    constraint_max=None,
                    constraint_message=""
                )
                st.session_state.min_frequency = int(min_frequency)

                min_percentage_range = (data['Percentage'].min(), data['Percentage'].max())
                init_percentage = float(min_percentage_range[0]) if st.session_state.min_percentage is None else st.session_state.min_percentage
                
                min_percentage = create_constrained_slider_with_input(
                    "Minimum Prevalence (%)",
                    float(min_percentage_range[0]), float(min_percentage_range[1]),
                    init_percentage,
                    0.1, "combinations_percentage",
                    "Minimum percentage of population affected",
                    is_float=True,
                    constraint_max=None,
                    constraint_message=""
                )
                st.session_state.min_percentage = min_percentage

                

                analyse_combinations_button = st.button(
                    "🔍 Analyse Combinations",
                    help="Click to analyse condition combinations"
                )
            except Exception as e:
                st.error(f"Error setting up analysis parameters: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

    with main_col:
        try:
            if analyse_combinations_button:
                with st.spinner("🔄 Analysing combinations..."):
                    # Clear previous results
                    st.session_state.combinations_results = None
                    st.session_state.combinations_fig = None
                    
                    # Use the actual widget values (these reflect current state including unfocused input)
                    current_min_frequency = min_frequency
                    current_min_percentage = min_percentage
                    
                    # Generate new results - use cross-population analysis for combined datasets
                    try:
                        results_df = analyze_condition_combinations(
                                data,
                                current_min_percentage,
                                current_min_frequency
                            )

                        if results_df.empty:
                            st.warning(f"No condition combinations found matching the filter criteria (Min Frequency: {current_min_frequency}, Min Prevalence: {current_min_percentage}%). Please try adjusting the filter values.")
                            return
                            
                    except Exception as analysis_error:
                        st.warning(f"No condition combinations found matching the filter criteria (Min Frequency: {current_min_frequency}, Min Prevalence: {current_min_percentage}%). Please try adjusting the filter values.")
                        return

                    # Process results
                    st.session_state.combinations_results = results_df
                    
                    st.subheader(f"Analysis Results ({len(results_df)} combinations)")
                    
                    # Apply appropriate styling based on dataset type
                    styled_df = results_df.style.background_gradient(
                        cmap='YlOrRd',
                        subset=['Prevalence % (Based on MPF)']
                    )
                    
                    st.dataframe(styled_df)

                    # Create plot with appropriate column based on dataset type
                    fig = create_combinations_plot(results_df)
                    
                    st.session_state.combinations_fig = fig
                    st.pyplot(fig)

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="condition_combinations.csv",
                        mime="text/csv"
                    )
            
            # Display existing results if available
            elif st.session_state.combinations_results is not None:
                results_df = st.session_state.combinations_results
                
                
                
                st.subheader(f"Analysis Results ({len(results_df)} combinations)")
                
                # Apply appropriate styling based on dataset type
                styled_df = results_df.style.background_gradient(
                    cmap='YlOrRd',
                    subset=['Prevalence % (Based on MPF)']
                )
                
                st.dataframe(styled_df)

                if st.session_state.combinations_fig is not None:
                    st.pyplot(st.session_state.combinations_fig)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="condition_combinations.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error in combinations analysis: {str(e)}")
            st.session_state.combinations_results = None
            st.session_state.combinations_fig = None

def render_personalised_analysis_tab(data):
    st.header("Personalised Trajectory Analysis")
    st.markdown("""
    Analyse potential disease progressions based on a patient's current conditions,
    considering population-level statistics and time-based progression patterns.
    """)

    main_col, control_col = st.columns([3, 1])

    with control_col:
        with st.container():
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            st.markdown("### Control Panel")
            
            # Get absolute min/max values from data
            absolute_min_or = float(data['OddsRatio'].min())
            absolute_max_or = float(data['OddsRatio'].max())
            absolute_max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(
                lambda x: parse_iqr(x)[0]).max())
            
            # Get unique conditions
            unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
            
            # Condition selection callback to reset filters
            def on_condition_select_personal():
                st.session_state.selected_conditions = st.session_state.personal_conditions_select
                newly_selected_conditions = st.session_state.personal_conditions_select

                # Reset filters to their minimums/defaults
                st.session_state.min_or = absolute_min_or
                st.session_state.time_horizon = float(absolute_max_years)
                st.session_state.time_margin = 0.10

                # Reset the underlying slider/input widget states
                if 'personal_min_or_slider' in st.session_state:
                    st.session_state.personal_min_or_slider = absolute_min_or
                    st.session_state.personal_min_or_input = absolute_min_or
                if 'personal_time_horizon_slider' in st.session_state:
                    st.session_state.personal_time_horizon_slider = float(absolute_max_years)
                    st.session_state.personal_time_horizon_input = float(absolute_max_years)
                if 'personal_time_margin_slider' in st.session_state:
                    st.session_state.personal_time_margin_slider = 0.10
                    st.session_state.personal_time_margin_input = 0.10

                # Clear any constraint messages
                for key in st.session_state:
                    if key.startswith('personal_') and key.endswith('_constraint_msg'):
                        st.session_state[key] = ""

            # Condition selection
            st.multiselect(
                "Select Current Conditions",
                options=unique_conditions,
                default=st.session_state.selected_conditions,
                key="personal_conditions_select",
                on_change=on_condition_select_personal,
                help="Choose all conditions that the patient currently has"
            )
            
            selected_conditions = st.session_state.selected_conditions
            
            # Show filters only after conditions are selected
            if selected_conditions:
                st.markdown("---")  # Visual separator
                
                # Get constraints based on selected conditions
                constraint_max_or, constraint_max_freq, constraint_max_time, condition_details = get_condition_constraints(
                    data, selected_conditions
                )
                
                # Create constraint messages
                or_constraint_msg = ""
                time_constraint_msg = ""
                if constraint_max_or is not None:
                    limiting_conditions = [cond for cond, details in condition_details.items() 
                                         if details['max_or'] == constraint_max_or]
                    or_constraint_msg = f"Maximum available for selected conditions ({', '.join(limiting_conditions)})"
                    
                if constraint_max_time is not None:
                    limiting_conditions = [cond for cond, details in condition_details.items() 
                                         if details['max_time'] == constraint_max_time]
                    time_constraint_msg = f"Maximum available for selected conditions ({', '.join(limiting_conditions)})"
                
                # Create constrained sliders
                min_or = create_constrained_slider_with_input(
                    "Minimum Odds Ratio",
                    absolute_min_or, absolute_max_or,
                    st.session_state.min_or,
                    0.1, "personal_min_or",
                    "Filter trajectories by minimum odds ratio",
                    is_float=True, show_tip=True,
                    constraint_max=constraint_max_or,
                    constraint_message=or_constraint_msg
                )
                st.session_state.min_or = min_or
                
                time_horizon = create_constrained_slider_with_input(
                    "Time Horizon (years)",
                    1.0, float(absolute_max_years),
                    float(st.session_state.time_horizon),
                    0.5, "personal_time_horizon",
                    "Maximum time period to consider",
                    is_float=True,
                    constraint_max=constraint_max_time,
                    constraint_message=time_constraint_msg
                )
                st.session_state.time_horizon = time_horizon
        
                time_margin = create_constrained_slider_with_input(
                    "Time Margin",
                    0.0, 0.5,
                    st.session_state.time_margin,
                    0.05, "personal_time_margin",
                    "Allowable variation in time predictions",
                    is_float=True,
                    constraint_max=None,
                    constraint_message=""
                )
                st.session_state.time_margin = time_margin
                
                analyse_button = st.button(
                    "🔍 Analyse Trajectories",
                    key="personal_analyse",
                    help="Generate personalised analysis"
                )
            else:
                analyse_button = False
                
            st.markdown('</div>', unsafe_allow_html=True)

    with main_col:
        selected_conditions = st.session_state.selected_conditions
        
        if selected_conditions and analyse_button:
            with st.spinner("🔄 Generating personalised analysis..."):
                # Generate new analysis
                html_content = create_personalized_analysis(
                    data,
                    selected_conditions,
                    time_horizon,
                    time_margin,
                    min_or
                )
                st.session_state.personalized_html = html_content

                html_container = f"""
                <div style="min-height: 800px; width: 100%; padding: 20px;">
                    {html_content}
                </div>
                """
                st.components.v1.html(html_container, height=1200, scrolling=True)
                st.download_button(
                    label="📥 Download Analysis",
                    data=html_content,
                    file_name="personalised_trajectory_analysis.html",
                    mime="text/html"
                )

        # Display existing analysis if available
        elif st.session_state.personalized_html is not None:
            html_container = f"""
            <div style="min-height: 800px; width: 100%; padding: 20px;">
                {st.session_state.personalized_html}
            </div>
            """
            st.components.v1.html(html_container, height=1200, scrolling=True)
            st.download_button(
                label="📥 Download Analysis",
                data=st.session_state.personalized_html,
                file_name="personalised_trajectory_analysis.html",
                mime="text/html"
            )

def calculate_min_required_filters(data, selected_conditions):
    """
    Calculates the minimum Odds Ratio, Minimum Frequency, and maximum Time Horizon
    required to include all selected conditions.
    """
    if not selected_conditions:
        return None, None, None, "" # No conditions selected, no constraints

    min_or_required = 0.0
    min_freq_required = 0
    max_time_horizon_required = 0.0

    # Keep track of conditions that couldn't be found in any trajectory
    unfound_conditions = []

    for cond in selected_conditions:
        # Find all trajectories where this condition is either ConditionA or ConditionB
        relevant_trajectories = data[
            (data['ConditionA'] == cond) | (data['ConditionB'] == cond)
        ]

        if relevant_trajectories.empty:
            unfound_conditions.append(cond)
            continue

        # Find the most permissive (lowest) OR and Freq for this condition
        # And the most permissive (highest) Time Horizon
        current_cond_min_or = relevant_trajectories['OddsRatio'].min()
        current_cond_min_freq = relevant_trajectories['PairFrequency'].min()
        current_cond_max_time_horizon = relevant_trajectories['MedianDurationYearsWithIQR'].apply(
            lambda x: parse_iqr(x)[0]).max()

        # Update overall required filters
        # For OR and Freq, we need the highest of the minimums (most restrictive)
        min_or_required = max(min_or_required, current_cond_min_or)
        min_freq_required = max(min_freq_required, current_cond_min_freq)
        # For Time Horizon, we need the highest of the maximums (most permissive)
        max_time_horizon_required = max(max_time_horizon_required, current_cond_max_time_horizon)

    message = ""
    if unfound_conditions:
        message = f"Note: The following selected conditions have no associated trajectories with current filters: {', '.join(unfound_conditions)}. They will not be displayed."

    return min_or_required, min_freq_required, max_time_horizon_required, message

def get_condition_constraints(data, selected_conditions):
    """
    Get the maximum available filter values for the selected conditions.
    Returns (max_min_or, max_min_freq, max_time_horizon, constraint_info)
    """
    if not selected_conditions:
        return None, None, None, {}
    
    max_min_or = 0.0
    max_min_freq = 0
    max_time_horizon = 0.0
    condition_details = {}
    
    for cond in selected_conditions:
        # Find all trajectories where this condition appears
        relevant_trajectories = data[
            (data['ConditionA'] == cond) | (data['ConditionB'] == cond)
        ]
        
        if not relevant_trajectories.empty:
            cond_max_min_or = relevant_trajectories['OddsRatio'].max()
            cond_max_min_freq = relevant_trajectories['PairFrequency'].max()
            cond_max_time_horizon = relevant_trajectories['MedianDurationYearsWithIQR'].apply(
                lambda x: parse_iqr(x)[0]).max()
            
            condition_details[cond] = {
                'max_or': cond_max_min_or,
                'max_freq': cond_max_min_freq,
                'max_time': cond_max_time_horizon
            }
            
            # Take the minimum of the maximums (most restrictive constraint)
            if max_min_or == 0.0:  # First condition
                max_min_or = cond_max_min_or
                max_min_freq = cond_max_min_freq
                max_time_horizon = cond_max_time_horizon
            else:
                max_min_or = min(max_min_or, cond_max_min_or)
                max_min_freq = min(max_min_freq, cond_max_min_freq)
                max_time_horizon = min(max_time_horizon, cond_max_time_horizon)
    
    return max_min_or, max_min_freq, max_time_horizon, condition_details

def create_constrained_slider_with_input(label, absolute_min, absolute_max, current_val, step, key_prefix, 
                                       help_text="", is_float=True, show_tip=False, 
                                       constraint_max=None, constraint_message=""):
    """
    Create a slider that snaps back to constraint_max if user tries to exceed it.
    """
    slider_key = f"{key_prefix}_slider"
    input_key = f"{key_prefix}_input"
    constraint_key = f"{key_prefix}_constraint_msg"
    
    # Initialize session state
    if slider_key not in st.session_state:
        st.session_state[slider_key] = max(absolute_min, min(absolute_max, current_val))
    if input_key not in st.session_state:
        st.session_state[input_key] = max(absolute_min, min(absolute_max, current_val))
    if constraint_key not in st.session_state:
        st.session_state[constraint_key] = ""
    
    # Determine effective max (constrained if conditions are selected)
    effective_max = constraint_max if constraint_max is not None else absolute_max
    
    def on_slider_change():
        attempted_value = st.session_state[slider_key]
        
        if constraint_max is not None and attempted_value > constraint_max:
            # Snap back to constraint max (rounded consistently)
            snapped_value = float(f"{constraint_max:.2f}") if is_float else int(constraint_max)
            st.session_state[slider_key] = snapped_value
            st.session_state[input_key] = snapped_value
            if is_float:
                rounded_str = f"{snapped_value:.2f}"
            else:
                rounded_str = str(snapped_value)
            st.session_state[constraint_key] = f"⚠️ Limited to {rounded_str}: {constraint_message}"
        else:
            # Normal behavior
            st.session_state[input_key] = max(absolute_min, min(effective_max, attempted_value))
            st.session_state[constraint_key] = ""
    
    def on_input_change():
        attempted_value = st.session_state[input_key]
        
        if attempted_value < absolute_min:
            # Snap to absolute minimum (rounded consistently)
            snapped_value = float(f"{absolute_min:.2f}") if is_float else int(absolute_min)
            st.session_state[slider_key] = snapped_value
            st.session_state[input_key] = snapped_value
            if is_float:
                min_str = f"{snapped_value:.2f}"
            else:
                min_str = str(snapped_value)
            st.session_state[constraint_key] = f"⚠️ Minimum value is {min_str}"
        elif attempted_value > absolute_max:
            # Snap to effective maximum (constraint max if available, otherwise absolute max)
            effective_snap_max = constraint_max if constraint_max is not None else absolute_max
            snapped_value = float(f"{effective_snap_max:.2f}") if is_float else int(effective_snap_max) 
            st.session_state[slider_key] = snapped_value
            st.session_state[input_key] = snapped_value
            if constraint_max is not None:
                if is_float:
                    rounded_str = f"{snapped_value:.2f}"
                else:
                    rounded_str = str(snapped_value)
                st.session_state[constraint_key] = f"⚠️ Limited to {rounded_str}: {constraint_message}"
            else:
                if is_float:
                    max_str = f"{snapped_value:.2f}"
                else:
                    max_str = str(snapped_value)
                st.session_state[constraint_key] = f"⚠️ Maximum value is {max_str}"
        elif constraint_max is not None and attempted_value > constraint_max:
            # Snap back to constraint max (rounded consistently)
            snapped_value = float(f"{constraint_max:.2f}") if is_float else int(constraint_max)
            st.session_state[slider_key] = snapped_value
            st.session_state[input_key] = snapped_value
            if is_float:
                rounded_str = f"{snapped_value:.2f}"
            else:
                rounded_str = str(snapped_value)
            st.session_state[constraint_key] = f"⚠️ Limited to {rounded_str}: {constraint_message}"
        else:
            # Normal behavior
            st.session_state[slider_key] = max(absolute_min, min(effective_max, attempted_value))
            st.session_state[constraint_key] = ""
    
    # Create columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        slider_val = st.slider(
            label,
            min_value=absolute_min,
            max_value=absolute_max,  # Keep absolute max for slider range
            step=step,
            key=slider_key,
            on_change=on_slider_change,
            help=help_text
        )
    
    with col2:
        input_val = st.number_input(
            "Or type:",
            step=step,
            key=input_key,
            on_change=on_input_change,
            help=f"Press Enter to register. {help_text}" if help_text else "Press Enter to register"
        )
        if show_tip:
            st.caption("💡 Press Enter after typing")
    
    # Show constraint message if present
    if st.session_state[constraint_key]:
        st.caption(st.session_state[constraint_key])
    
    return input_val

def render_trajectory_filter_tab(data):
    st.header("Custom Trajectory Filter")
    st.markdown("""
    Visualise disease trajectories based on custom odds ratio and frequency thresholds.
    Select conditions and adjust filters to explore different trajectory patterns.
    """)

    # Initialize session state
    if 'custom_filter_reverted_message' not in st.session_state:
        st.session_state.custom_filter_reverted_message = ""
    if 'selected_conditions' not in st.session_state:
        st.session_state.selected_conditions = []

    # Get total unique conditions from the unfiltered data
    total_unique_conditions_unfiltered = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))

    main_col, control_col = st.columns([3, 1])

    with control_col:
        with st.container():
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            st.markdown("### Control Panel")
            
            # Initialize button state
            generate_button = False
            
            try:
                # Get absolute min/max values from data for resetting and slider ranges
                absolute_min_or = float(data['OddsRatio'].min())
                absolute_max_or = float(data['OddsRatio'].max())
                absolute_min_freq = int(data['PairFrequency'].min())
                absolute_max_freq = int(data['PairFrequency'].max())
                absolute_max_years = math.ceil(data['MedianDurationYearsWithIQR']
                                        .apply(lambda x: parse_iqr(x)[0]).max())

                # Condition selection callback to reset filters
                def on_condition_select_callback():
                    st.session_state.selected_conditions = st.session_state.custom_select
                    newly_selected_conditions = st.session_state.custom_select

                    # Reset OR and Freq filters to their minimums
                    st.session_state.min_or = absolute_min_or
                    st.session_state.min_freq = absolute_min_freq

                    # Calculate constraints to find the new default for Time Horizon
                    _, _, constraint_max_time, _ = get_condition_constraints(
                        data, newly_selected_conditions
                    )

                    # Set new defaults for Time Horizon and Margin
                    new_time_horizon_default = constraint_max_time if constraint_max_time and constraint_max_time > 0 else absolute_max_years
                    new_time_margin_default = 0.10

                    st.session_state.time_horizon = new_time_horizon_default
                    st.session_state.time_margin = new_time_margin_default

                    # Reset the underlying slider/input widget states
                    if 'custom_min_or_slider' in st.session_state:
                        st.session_state.custom_min_or_slider = absolute_min_or
                        st.session_state.custom_min_or_input = absolute_min_or
                    if 'custom_min_freq_slider' in st.session_state:
                        st.session_state.custom_min_freq_slider = absolute_min_freq
                        st.session_state.custom_min_freq_input = absolute_min_freq
                    if 'custom_time_horizon_slider' in st.session_state:
                        st.session_state.custom_time_horizon_slider = new_time_horizon_default
                        st.session_state.custom_time_horizon_input = new_time_horizon_default
                    if 'custom_time_margin_slider' in st.session_state:
                        st.session_state.custom_time_margin_slider = new_time_margin_default
                        st.session_state.custom_time_margin_input = new_time_margin_default

                    # Clear any constraint messages
                    for key in st.session_state:
                        if key.endswith('_constraint_msg'):
                            st.session_state[key] = ""

                # --- Step 1: Select Conditions ---
                st.multiselect(
                    "Select Initial Conditions",
                    options=total_unique_conditions_unfiltered,
                    default=st.session_state.selected_conditions,
                    key="custom_select",
                    on_change=on_condition_select_callback,
                    help="Choose conditions to begin. Filters will appear after selection."
                )
                
                selected_conditions = st.session_state.selected_conditions

                # --- Step 2: Show filters only after conditions are selected ---
                if selected_conditions:
                    st.markdown("---") # Visual separator
                    
                    # Get constraints based on selected conditions
                    constraint_max_or, constraint_max_freq, constraint_max_time, condition_details = get_condition_constraints(
                        data, selected_conditions
                    )
                    
                    # Create constraint messages
                    or_constraint_msg = ""
                    freq_constraint_msg = ""
                    time_constraint_msg = ""
                    if constraint_max_or is not None:
                        limiting_conditions = [cond for cond, details in condition_details.items() 
                                             if details['max_or'] == constraint_max_or]
                        or_constraint_msg = f"Maximum available for selected conditions ({', '.join(limiting_conditions)})"
                        
                    if constraint_max_freq is not None:
                        limiting_conditions = [cond for cond, details in condition_details.items() 
                                             if details['max_freq'] == constraint_max_freq]
                        freq_constraint_msg = f"Maximum available for selected conditions ({', '.join(limiting_conditions)})"
                        
                    if constraint_max_time is not None:
                        limiting_conditions = [cond for cond, details in condition_details.items() 
                                             if details['max_time'] == constraint_max_time]
                        time_constraint_msg = f"Maximum available for selected conditions ({', '.join(limiting_conditions)})"

                    # Create constrained sliders
                    min_or = create_constrained_slider_with_input(
                        "Minimum Odds Ratio",
                        absolute_min_or, absolute_max_or,
                        st.session_state.min_or,
                        0.1, "custom_min_or",
                        "Filter trajectories by minimum odds ratio",
                        is_float=True, show_tip=True,
                        constraint_max=constraint_max_or,
                        constraint_message=or_constraint_msg
                    )

                    min_freq = create_constrained_slider_with_input(
                        "Minimum Frequency",
                        absolute_min_freq, absolute_max_freq,
                        getattr(st.session_state, 'min_freq', absolute_min_freq),
                        1, "custom_min_freq",
                        "Minimum number of occurrences required",
                        is_float=False,
                        constraint_max=constraint_max_freq,
                        constraint_message=freq_constraint_msg
                    )
                    
                    time_horizon = create_constrained_slider_with_input(
                        "Time Horizon (years)",
                        1.0, float(absolute_max_years),
                        float(st.session_state.time_horizon),
                        0.5, "custom_time_horizon",
                        "Maximum time period to consider",
                        is_float=True,
                        constraint_max=constraint_max_time,
                        constraint_message=time_constraint_msg
                    )

                    time_margin = create_constrained_slider_with_input(
                        "Time Margin",
                        0.0, 0.5,
                        st.session_state.time_margin,
                        0.05, "custom_time_margin",
                        "Allowable variation in time predictions",
                        is_float=True,
                        constraint_max=None,
                        constraint_message=""
                    )
                    
                    # Update session state
                    st.session_state.min_or = min_or
                    st.session_state.min_freq = min_freq
                    st.session_state.time_horizon = time_horizon
                    st.session_state.time_margin = time_margin

                    # Generate button
                    generate_button = st.button(
                        "🔄 Generate Network",
                        key="custom_generate",
                        help="Click to generate trajectory network"
                    )

            except Exception as e:
                st.error(f"Error in custom trajectory analysis: {str(e)}")
                
            st.markdown('</div>', unsafe_allow_html=True)

    with main_col:
        if selected_conditions and generate_button:
            with st.spinner("🌐 Generating network..."):
                try:
                    # Fetch current filter values from session state for this run
                    min_or_val = st.session_state.min_or
                    min_freq_val = st.session_state.min_freq
                    time_horizon_val = st.session_state.time_horizon
                    time_margin_val = st.session_state.time_margin

                    # Filter data based on the active filters
                    filtered_data = data[
                        (data['OddsRatio'] >= min_or_val) &
                        (data['PairFrequency'] >= min_freq_val)
                    ]
                    
                    # Clear previous network
                    st.session_state.network_html = None
                    
                    # Generate new network with current values
                    html_content = create_network_graph(
                        filtered_data,
                        selected_conditions,
                        min_or_val,
                        time_horizon_val,
                        time_margin_val
                    )
                    st.session_state.network_html = html_content
                    st.components.v1.html(html_content, height=800)

                    st.download_button(
                        label="📥 Download Network",
                        data=html_content,
                        file_name="custom_trajectory_network.html",
                        mime="text/html"
                    )
                except Exception as viz_error:
                    st.error(f"Failed to generate network visualisation: {str(viz_error)}")
                    st.session_state.network_html = None

        # Display existing network if available
        elif st.session_state.network_html is not None:
            st.components.v1.html(st.session_state.network_html, height=800)
            st.download_button(
                label="📥 Download Network",
                data=st.session_state.network_html,
                file_name="custom_trajectory_network.html",
                mime="text/html"
            )

def render_cohort_network_tab(data):
    st.header("Cohort Network Analysis")
    st.markdown("""
    Visualize relationships between conditions as a network graph. 
    Node colors represent body systems, and edge thickness indicates association strength.
    """)

    main_col, control_col = st.columns([3, 1])

    with control_col:
        with st.container():
            st.markdown("### Control Panel")
            
            # Dynamically set slider ranges based on the loaded data
            min_or_range = (data['OddsRatio'].min(), data['OddsRatio'].max())
            min_freq_range = (data['PairFrequency'].min(), data['PairFrequency'].max())
            
            # Sliders for filtering
            min_or = create_constrained_slider_with_input(
                "Minimum Odds Ratio",
                float(min_or_range[0]), float(min_or_range[1]),
                2.0,
                0.1, "cohort_network_min_or",
                "Filter relationships by minimum odds ratio",
                is_float=True, show_tip=True,
                constraint_max=None,
                constraint_message=""
            )

            min_freq = create_constrained_slider_with_input(
                "Minimum Pair Frequency",
                int(min_freq_range[0]), int(min_freq_range[1]),
                int(min_freq_range[0]),
                1, "cohort_network_min_freq",
                "Minimum number of occurrences required",
                is_float=False,
                constraint_max=None,
                constraint_message=""
            )

            generate_button = st.button(
                "🔄 Generate Network",
                key="cohort_network_generate",
                help="Create network visualization"
            )

    with main_col:
        if generate_button:
            with st.spinner("🌐 Generating network visualization..."):
                try:
                    # Calculate summary statistics
                    filtered_data = data[
                        (data['OddsRatio'] >= min_or) &
                        (data['PairFrequency'] >= min_freq)
                    ]

                    # Add a check to ensure filtered_data is not empty
                    if filtered_data.empty:
                        st.warning("No data matches the current filter criteria. Please adjust the sliders.")
                        return
                    
                    # Create visualization
                    html_content = create_network_visualization(filtered_data, min_or, min_freq)
                    
                    # Display network
                    st.components.v1.html(html_content, height=800)
                    
                    # Add download button
                    st.download_button(
                        label="📥 Download Network Visualization",
                        data=html_content,
                        file_name="condition_network.html",
                        mime="text/html"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating network visualization: {str(e)}")
