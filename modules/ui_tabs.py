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
            top_n = st.slider(
                "Number of Top Trajectories",
                min_value=1,
                max_value=20,
                value=st.session_state.top_n_trajectories,
                step=1,
                help="Select how many top trajectories to display"
            )
            st.session_state.top_n_trajectories = top_n
            
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
                    st.pyplot(fig)

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
                st.pyplot(fig)

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
                min_frequency = st.slider(
                    "Minimum Pair Frequency",
                    int(min_freq_range[0]),
                    int(min_freq_range[1]),
                    int(min_freq_range[0]) if st.session_state.min_frequency is None
                    else st.session_state.min_frequency,
                    help="Minimum number of occurrences required"
                )
                st.session_state.min_frequency = min_frequency

                min_percentage_range = (data['Percentage'].min(), data['Percentage'].max())
                min_percentage = st.slider(
                    "Minimum Prevalence (%)",
                    float(min_percentage_range[0]),
                    float(min_percentage_range[1]),
                    float(min_percentage_range[0]) if st.session_state.min_percentage is None
                    else st.session_state.min_percentage,
                    0.1,
                    help="Minimum percentage of population affected"
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
                    
                    # Generate new results - use cross-population analysis for combined datasets
                    results_df = analyze_condition_combinations(
                            data,
                            min_percentage,
                            min_frequency
                        )

                    if not results_df.empty:
                        st.session_state.combinations_results = results_df
                        
                        
                        
                        st.subheader(f"Analysis Results ({len(results_df)} combinations)")
                        
                        # Apply appropriate styling based on dataset type
                        styled_df = results_df.style.background_gradient(
                            cmap='YlOrRd',
                            subset=['Prevalence of the combination (%)']
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
                    else:
                        st.warning("No combinations found matching the criteria. Try adjusting the parameters.")
            
            # Display existing results if available
            elif st.session_state.combinations_results is not None:
                results_df = st.session_state.combinations_results
                
                
                
                st.subheader(f"Analysis Results ({len(results_df)} combinations)")
                
                # Apply appropriate styling based on dataset type
                styled_df = results_df.style.background_gradient(
                    cmap='YlOrRd',
                    subset=['Prevalence of the combination (%)']
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
            
            # Get min/max values from data
            min_or_value = float(data['OddsRatio'].min())
            max_or_value = float(data['OddsRatio'].max())
            
            min_or = st.slider(
                "Minimum Odds Ratio",
                min_value=min_or_value,
                max_value=max_or_value,
                value=st.session_state.min_or,
                step=0.5,
                key="personal_min_or",
                help="Filter trajectories by minimum odds ratio"
            )
            st.session_state.min_or = min_or

            # Get max years from data
            max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(
                lambda x: parse_iqr(x)[0]).max())
            
            time_horizon = st.slider(
                "Time Horizon (years)",
                min_value=1.0,
                max_value=float(max_years),
                value=float(st.session_state.time_horizon),
                step=0.5,
                key="personal_time_horizon",
                help="Maximum time period to consider"
            )
            st.session_state.time_horizon = time_horizon
    
            time_margin = st.slider(
                "Time Margin",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.time_margin,
                step=0.05,
                key="personal_time_margin",
                help="Allowable variation in time predictions"
            )
            st.session_state.time_margin = time_margin
            
            analyse_button = st.button(
                "🔍 Analyse Trajectories",
                key="personal_analyse",
                help="Generate personalised analysis"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with main_col:
        st.markdown("""
        <h4 style='font-size: 1.2em; font-weight: 600; color: #333; margin-bottom: 10px;'>
            🔍 Please select all conditions that the patient currently has:
        </h4>
        """, unsafe_allow_html=True)
        
        # Initialize session state for selected conditions if not exists
        if 'selected_conditions' not in st.session_state:
            st.session_state.selected_conditions = []

        # Get unique conditions only once
        unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
        
        def on_condition_select():
            # Update the session state directly from the widget value
            st.session_state.selected_conditions = st.session_state.personal_select

        # Use the multiselect with a callback
        selected_conditions = st.multiselect(
            "Select Current Conditions",
            options=unique_conditions,
            default=st.session_state.selected_conditions,
            key="personal_select",
            on_change=on_condition_select,
            help="Choose all conditions that the patient currently has"
        )

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

def render_trajectory_filter_tab(data):
    st.header("Custom Trajectory Filter")
    st.markdown("""
    Visualise disease trajectories based on custom odds ratio and frequency thresholds.
    Select conditions and adjust filters to explore different trajectory patterns.
    """)

    main_col, control_col = st.columns([3, 1])

    with control_col:
        with st.container():
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            st.markdown("### Control Panel")
            try:
                # Get min/max values from data
                min_or_value = float(data['OddsRatio'].min())
                max_or_value = float(data['OddsRatio'].max())
                min_freq_value = int(data['PairFrequency'].min())
                max_freq_value = int(data['PairFrequency'].max())
                
                min_or = st.slider(
                    "Minimum Odds Ratio",
                    min_value=min_or_value,
                    max_value=max_or_value,
                    value=st.session_state.min_or,
                    step=0.5,
                    key="custom_min_or",
                    help="Filter trajectories by minimum odds ratio"
                )

                min_freq = st.slider(
                    "Minimum Frequency",
                    min_value=min_freq_value,
                    max_value=max_freq_value,
                    value=min_freq_value,
                    step=1,
                    help="Minimum number of occurrences required"
                )

                # Filter data based on both OR and frequency
                filtered_data = data[
                    (data['OddsRatio'] >= min_or) &
                    (data['PairFrequency'] >= min_freq)
                ]

                # Get conditions from filtered data
                unique_conditions = sorted(set(
                    filtered_data['ConditionA'].unique()) |
                    set(filtered_data['ConditionB'].unique())
                )

                # Use the same session state as tab 2
                if 'selected_conditions' not in st.session_state:
                    st.session_state.selected_conditions = []

                def on_condition_select():
                    # Update the shared session state
                    st.session_state.selected_conditions = st.session_state.custom_select

                selected_conditions = st.multiselect(
                    "Select Initial Conditions",
                    options=unique_conditions,
                    default=st.session_state.selected_conditions,
                    key="custom_select",
                    on_change=on_condition_select,
                    help="Choose the starting conditions for trajectory analysis"
                )

        
                if selected_conditions:
                    max_years = math.ceil(filtered_data['MedianDurationYearsWithIQR']
                                    .apply(lambda x: parse_iqr(x)[0]).max())
                    time_horizon = st.slider(
                        "Time Horizon (years)",
                        min_value=1.0,
                        max_value=float(max_years),  # Convert to float
                        value=float(st.session_state.time_horizon),  # Convert to float
                        step=0.5,  
                        key="custom_time_horizon",
                        help="Maximum time period to consider"
                    )


                    time_margin = st.slider(
                        "Time Margin",
                        0.0, 0.5, st.session_state.time_margin, 0.05,
                        key="custom_time_margin",
                        help="Allowable variation in time predictions"
                    )

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
                    # Clear previous network
                    st.session_state.network_html = None
                    
                    # Generate new network
                    html_content = create_network_graph(
                        filtered_data,
                        selected_conditions,
                        min_or,
                        time_horizon,
                        time_margin
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
            min_or = st.slider(
                "Minimum Odds Ratio",
                float(min_or_range[0]), float(min_or_range[1]), 2.0, 0.1,
                key="cohort_network_min_or",
                help="Filter relationships by minimum odds ratio"
            )

            min_freq = st.slider(
                "Minimum Pair Frequency",
                int(min_freq_range[0]),
                int(min_freq_range[1]),
                int(min_freq_range[0]),
                key="cohort_network_min_freq",
                help="Minimum number of occurrences required"
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
