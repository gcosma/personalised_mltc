with tab2:
    st.header("Trajectory Prediction")

    # Create columns with the trajectory area taking up more space
    col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed (3:1 means main area is 3 times larger)

    with col1:
        # Main area for condition selection and network display
        unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
        selected_conditions = st.multiselect("Select Initial Conditions", unique_conditions)

        # Placeholder for network visualization
        network_placeholder = st.empty()

    with col2:
        # Sidebar-like column for parameters
        st.subheader("Network Parameters")
        min_or = st.slider("Minimum Odds Ratio", 1.0, 10.0, 2.0, 0.5)
        
        if selected_conditions:
            max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(lambda x: parse_iqr(x)[0]).max())
            time_horizon = st.slider("Time Horizon (years)", 1, max_years, min(5, max_years))
            time_margin = st.slider("Time Margin", 0.0, 0.5, 0.1, 0.05)

            # Generate button aligned to the right
            generate_button = st.button("Generate Trajectory Network")

    # Network generation logic
    if selected_conditions and generate_button:
        with st.spinner("Generating network visualization..."):
            try:
                # Directly generate HTML content
                html_content = create_network_graph(
                    data, 
                    selected_conditions, 
                    min_or, 
                    time_horizon, 
                    time_margin
                )
                
                # Display network in the main area placeholder
                network_placeholder.components.v1.html(html_content, height=800)
                
                # Download button can remain in the parameters column
                st.download_button(
                    label="Download Network Graph",
                    data=html_content,
                    file_name="trajectory_network.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Failed to generate network graph: {e}")
