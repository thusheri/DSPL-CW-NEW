import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Money Supply and Economic Impact Dashboard",
    page_icon="💰",
    layout="wide"
)

# Create custom CSS styling
st.markdown("""
<style>
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        flex: 1;
        min-width: 200px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-title {
        font-size: 1rem;
        color: #4B5563;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv('Cleaned_dataset.csv', encoding='utf-8')  # or 'utf8'
    

    # Extract date information
    df['Date'] = df['End of the Year'].str.strip()
    df[['Year', 'Month']] = df['Date'].str.extract(r'(\d{4})\s+(\w+)')
    
    # Convert columns to numeric, handling non-numeric values
    numeric_cols = ['Currency Issues of the CBSL', 'Government Agencies Deposit with CBSL', 
                    'Commercial Bank Deposit with CBSL', 'Total', 'M1', 'M2', 'M2b', 
                    'M2 velocity', 'M2b velocity']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create a proper datetime column for sorting
    month_map = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    
    df['month_num'] = df['Month'].map(month_map)
    df['datetime'] = pd.to_datetime(df['Year'] + '-' + df['month_num'], format='%Y-%m', errors='coerce')
    
    # Sort by date
    df = df.sort_values('datetime')
    
    return df

# Load data
df = load_data()

# Title and introduction
st.markdown('<div class="header">Money Supply and Economic Impact Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
This dashboard analyzes the relationship between money supply metrics (M1, M2, M2b) and 
their velocities over time, exploring their potential impacts on the economy.
""")

# Sidebar for filtering
st.sidebar.title("Filters")

# Year range filter
years = sorted(df['Year'].unique())
start_year, end_year = st.sidebar.select_slider(
    "Select Year Range",
    options=years,
    value=(min(years), max(years))
)

# Month filter
all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
selected_months = st.sidebar.multiselect(
    "Select Months",
    options=all_months,
    default=all_months
)

# Filter the data based on selections
filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
if selected_months:
    filtered_df = filtered_df[filtered_df['Month'].isin(selected_months)]

# Sort filtered data by datetime
filtered_df = filtered_df.sort_values('datetime')

# Check if we have data after filtering
if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
else:
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Money Supply Overview", "Velocity Analysis", "Correlation Analysis", "Economic Implications"])
    
    # Tab 1: Money Supply Overview
    with tab1:
        st.markdown('<div class="subheader">Money Supply Trends</div>', unsafe_allow_html=True)
        
        # Show key metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        latest = filtered_df.iloc[-1]
        earliest = filtered_df.iloc[0]
        
        # Calculate growth rates for key metrics
        for metric in ['M1', 'M2', 'M2b']:
            if not pd.isna(latest[metric]) and not pd.isna(earliest[metric]) and earliest[metric] != 0:
                growth = ((latest[metric] - earliest[metric]) / earliest[metric]) * 100
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{metric} Growth</div>
                    <div class="metric-value">{growth:.2f}%</div>
                    <small>From {earliest['Date']} to {latest['Date']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Line chart for Money Supply over time
        st.subheader("Money Supply Trends")
        
        money_supply_fig = go.Figure()
        
        # Add traces for different money supply metrics
        for metric in ['M1', 'M2', 'M2b']:
            valid_data = filtered_df[~filtered_df[metric].isna()]
            money_supply_fig.add_trace(go.Scatter(
                x=valid_data['datetime'],
                y=valid_data[metric],
                name=metric,
                mode='lines+markers'
            ))
        
        money_supply_fig.update_layout(
            title="Money Supply Metrics Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Metric",
            hovermode="x unified"
        )
        
        st.plotly_chart(money_supply_fig, use_container_width=True)
        
        # Components of money supply
        st.subheader("Components of Money Supply")
        
        # Create a stacked area chart for the components
        components_fig = go.Figure()
        
        components = ['Currency Issues of the CBSL', 'Government Agencies Deposit with CBSL', 'Commercial Bank Deposit with CBSL']
        
        for component in components:
            valid_data = filtered_df[~filtered_df[component].isna()]
            components_fig.add_trace(go.Scatter(
                x=valid_data['datetime'],
                y=valid_data[component],
                name=component,
                mode='none',
                stackgroup='one'
            ))
        
        components_fig.update_layout(
            title="Components of Money Supply",
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Component",
            hovermode="x unified"
        )
        
        st.plotly_chart(components_fig, use_container_width=True)

    # Tab 2: Velocity Analysis
    with tab2:
        st.markdown('<div class="subheader">Money Velocity Analysis</div>', unsafe_allow_html=True)
        
        # Key velocity metrics
        velocity_cols = ['M2 velocity', 'M2b velocity']
        
        # Only show metrics if data exists
        valid_vel_df = filtered_df[velocity_cols].dropna(how='all')
        
        if not valid_vel_df.empty:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            
            latest_vel = filtered_df[velocity_cols].dropna(how='all').iloc[-1]
            earliest_vel = filtered_df[velocity_cols].dropna(how='all').iloc[0]
            
            for col in velocity_cols:
                if col in latest_vel and col in earliest_vel and not pd.isna(latest_vel[col]) and not pd.isna(earliest_vel[col]) and earliest_vel[col] != 0:
                    change = ((latest_vel[col] - earliest_vel[col]) / earliest_vel[col]) * 100
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{col} Change</div>
                        <div class="metric-value">{change:.2f}%</div>
                        <small>From {earliest_vel.name} to {latest_vel.name}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Line chart for velocity over time
            velocity_fig = go.Figure()
            
            for vel in velocity_cols:
                valid_data = filtered_df[~filtered_df[vel].isna()]
                if not valid_data.empty:
                    velocity_fig.add_trace(go.Scatter(
                        x=valid_data['datetime'],
                        y=valid_data[vel],
                        name=vel,
                        mode='lines+markers'
                    ))
            
            velocity_fig.update_layout(
                title="Money Velocity Over Time",
                xaxis_title="Date",
                yaxis_title="Velocity",
                legend_title="Metric",
                hovermode="x unified"
            )
            
            st.plotly_chart(velocity_fig, use_container_width=True)
            
            # Money Supply vs Velocity Analysis
            st.subheader("Money Supply vs Velocity Analysis")
            
            # Create dropdown for selecting which metrics to compare
            col1, col2 = st.columns(2)
            
            with col1:
                supply_metric = st.selectbox(
                    "Select Money Supply Metric",
                    options=['M1', 'M2', 'M2b'],
                    index=1  # Default to M2
                )
            
            with col2:
                velocity_metric = st.selectbox(
                    "Select Velocity Metric",
                    options=velocity_cols,
                    index=0  # Default to M2 velocity
                )
            
            # Create dual-axis chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add money supply trace
            valid_supply = filtered_df[~filtered_df[supply_metric].isna()]
            fig.add_trace(
                go.Scatter(x=valid_supply['datetime'], y=valid_supply[supply_metric], name=supply_metric),
                secondary_y=False,
            )
            
            # Add velocity trace
            valid_velocity = filtered_df[~filtered_df[velocity_metric].isna()]
            fig.add_trace(
                go.Scatter(x=valid_velocity['datetime'], y=valid_velocity[velocity_metric], name=velocity_metric),
                secondary_y=True,
            )
            
            # Set titles
            fig.update_layout(
                title_text="Money Supply vs. Velocity Over Time",
                hovermode="x unified"
            )
            
            # Set x-axis title
            fig.update_xaxes(title_text="Date")
            
            # Set y-axes titles
            fig.update_yaxes(title_text=supply_metric, secondary_y=False)
            fig.update_yaxes(title_text=velocity_metric, secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No velocity data available for the selected filters.")

    # Tab 3: Correlation Analysis
    with tab3:
        st.markdown('<div class="subheader">Correlation Analysis</div>', unsafe_allow_html=True)
        
        # Select variables for correlation
        st.subheader("Select Variables for Correlation Analysis")
        
        # Group columns logically
        money_supply_cols = ['M1', 'M2', 'M2b', 'Currency Issues of the CBSL', 
                            'Government Agencies Deposit with CBSL', 'Commercial Bank Deposit with CBSL', 'Total']
        velocity_cols = ['M2 velocity', 'M2b velocity']
        
        # Let user select variables
        cols_to_correlate = st.multiselect(
            "Select variables to include in correlation matrix",
            options=money_supply_cols + velocity_cols,
            default=['M1', 'M2', 'M2b', 'M2 velocity', 'M2b velocity']
        )
        
        if len(cols_to_correlate) > 1:
            # Calculate correlation
            corr_df = filtered_df[cols_to_correlate].corr()
            
            # Plot correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(
                corr_df, 
                mask=mask, 
                cmap=cmap, 
                vmax=1, 
                vmin=-1, 
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .5},
                annot=True,
                fmt=".2f"
            )
            
            plt.title("Correlation Matrix Between Selected Variables", fontsize=16)
            st.pyplot(fig)
            
            # Scatter plots for selected pairs
            st.subheader("Scatter Plot Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("Select X variable", options=cols_to_correlate, index=0)
            
            with col2:
                y_var = st.selectbox("Select Y variable", options=cols_to_correlate, index=min(1, len(cols_to_correlate)-1))
            
            # Create scatter plot
            valid_scatter = filtered_df[~filtered_df[x_var].isna() & ~filtered_df[y_var].isna()]
            
            scatter_fig = px.scatter(
                valid_scatter, 
                x=x_var, 
                y=y_var,
                trendline="ols",
                hover_data=["datetime", "Year", "Month"],
                title=f"Relationship between {x_var} and {y_var}"
            )
            
            # Add year labels to points
            scatter_fig.update_traces(
                text=valid_scatter['Year'],
                textposition='top center'
            )
            
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Calculate and display the correlation coefficient
            corr_value = valid_scatter[[x_var, y_var]].corr().iloc[0, 1]
            
            st.markdown(f"""
            <div class="card">
                <strong>Correlation Coefficient: {corr_value:.4f}</strong><br>
                <small>
                    A value close to 1 indicates a strong positive correlation.<br>
                    A value close to -1 indicates a strong negative correlation.<br>
                    A value close to 0 indicates little to no linear relationship.
                </small>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning("Please select at least two variables for correlation analysis.")

    # Tab 4: Economic Implications
    with tab4:
        st.markdown('<div class="subheader">Economic Implications</div>', unsafe_allow_html=True)
        
        # Calculate annual growth rates
        december_data = filtered_df[filtered_df['Month'] == 'Dec'].copy()
        
        if len(december_data) > 1:
            december_data = december_data.sort_values('Year')
            december_data['M2_Growth'] = december_data['M2'].pct_change() * 100
            december_data['M2b_Growth'] = december_data['M2b'].pct_change() * 100
            december_data['M2_Velocity_Change'] = december_data['M2 velocity'].pct_change() * 100
            december_data['M2b_Velocity_Change'] = december_data['M2b velocity'].pct_change() * 100
            
            # Visualize annual growth rates
            growth_fig = go.Figure()
            
            growth_metrics = ['M2_Growth', 'M2b_Growth']
            for metric in growth_metrics:
                valid_growth = december_data[~december_data[metric].isna()]
                if not valid_growth.empty:
                    growth_fig.add_trace(go.Bar(
                        x=valid_growth['Year'],
                        y=valid_growth[metric],
                        name=metric.replace('_', ' ')
                    ))
            
            growth_fig.update_layout(
                title="Annual Money Supply Growth Rates (December to December)",
                xaxis_title="Year",
                yaxis_title="Growth Rate (%)",
                legend_title="Metric",
                barmode='group'
            )
            
            st.plotly_chart(growth_fig, use_container_width=True)
            
            # Visualization for velocity changes
            velocity_change_fig = go.Figure()
            
            vel_change_metrics = ['M2_Velocity_Change', 'M2b_Velocity_Change']
            for metric in vel_change_metrics:
                valid_vel_change = december_data[~december_data[metric].isna()]
                if not valid_vel_change.empty:
                    velocity_change_fig.add_trace(go.Bar(
                        x=valid_vel_change['Year'],
                        y=valid_vel_change[metric],
                        name=metric.replace('_', ' ')
                    ))
            
            velocity_change_fig.update_layout(
                title="Annual Money Velocity Changes (December to December)",
                xaxis_title="Year",
                yaxis_title="Change (%)",
                legend_title="Metric",
                barmode='group'
            )
            
            st.plotly_chart(velocity_change_fig, use_container_width=True)
            
            # Implications section with insights
            st.subheader("Economic Implications of Money Supply and Velocity Changes")
            
            st.markdown("""
            <div class="card">
                <h3>Understanding the Impact</h3>
                <p>
                    Changes in money supply and velocity have significant implications for the economy:
                </p>
                <ul>
                    <li><strong>Inflation/Deflation Effects:</strong> Increases in money supply without corresponding increases in output can lead to inflation. Conversely, decreases in money supply or velocity can contribute to deflationary pressures.</li>
                    <li><strong>Economic Growth:</strong> Moderate growth in money supply typically supports economic growth by facilitating transactions and investment. However, excessive money supply growth may lead to economic instability.</li>
                    <li><strong>Monetary Policy Effectiveness:</strong> The relationship between money supply and velocity influences the effectiveness of monetary policy. When velocity is declining, increases in money supply may not sufficiently stimulate economic activity.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate average values across different time periods
            time_periods = {}
            for year in sorted(december_data['Year'].unique()):
                decade = int(year) // 10 * 10
                period = f"{decade}s"
                if period not in time_periods:
                    time_periods[period] = []
                time_periods[period].append(year)
            
            # Calculate average values for each period
            period_data = []
            for period, years in time_periods.items():
                period_df = december_data[december_data['Year'].isin([str(y) for y in years])]
                avg_m2 = period_df['M2'].mean()
                avg_m2_vel = period_df['M2 velocity'].mean()
                avg_m2b = period_df['M2b'].mean()
                avg_m2b_vel = period_df['M2b velocity'].mean()
                
                period_data.append({
                    'Period': period,
                    'Avg M2': avg_m2,
                    'Avg M2 Velocity': avg_m2_vel,
                    'Avg M2b': avg_m2b,
                    'Avg M2b Velocity': avg_m2b_vel
                })
            
            period_df = pd.DataFrame(period_data)
            
            # Display the period data
            if not period_df.empty:
                st.subheader("Average Values by Decade")
                st.dataframe(
                    period_df.style.format({
                        'Avg M2': '{:.2f}',
                        'Avg M2 Velocity': '{:.2f}',
                        'Avg M2b': '{:.2f}',
                        'Avg M2b Velocity': '{:.2f}'
                    }),
                    hide_index=True
                )
                
                # Create a visualization for decade averages
                decade_fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add traces for money supply
                decade_fig.add_trace(
                    go.Bar(x=period_df['Period'], y=period_df['Avg M2'], name='Avg M2'),
                    secondary_y=False,
                )
                
                decade_fig.add_trace(
                    go.Bar(x=period_df['Period'], y=period_df['Avg M2b'], name='Avg M2b'),
                    secondary_y=False,
                )
                
                # Add traces for velocity
                decade_fig.add_trace(
                    go.Scatter(x=period_df['Period'], y=period_df['Avg M2 Velocity'], 
                             name='Avg M2 Velocity', mode='lines+markers'),
                    secondary_y=True,
                )
                
                decade_fig.add_trace(
                    go.Scatter(x=period_df['Period'], y=period_df['Avg M2b Velocity'], 
                             name='Avg M2b Velocity', mode='lines+markers'),
                    secondary_y=True,
                )
                
                decade_fig.update_layout(
                    title='Money Supply and Velocity Trends by Decade',
                    barmode='group'
                )
                
                decade_fig.update_yaxes(title_text="Money Supply", secondary_y=False)
                decade_fig.update_yaxes(title_text="Velocity", secondary_y=True)
                
                st.plotly_chart(decade_fig, use_container_width=True)
            
            # Add conclusions and interpretations
            st.subheader("Key Insights")
            
            # Calculate some insights from the data for automatic commentary
            recent_years = december_data.tail(5)
            m2_trend = "increasing" if recent_years['M2'].iloc[-1] > recent_years['M2'].iloc[0] else "decreasing"
            vel_trend = "increasing" if recent_years['M2 velocity'].iloc[-1] > recent_years['M2 velocity'].iloc[0] else "decreasing"
            
            st.markdown(f"""
            <div class="card">
                <p>
                    Based on the analysis of money supply and velocity data, the following insights can be drawn:
                </p>
                <ul>
                    <li>In recent years, the money supply (M2) has been <strong>{m2_trend}</strong>, while money velocity has been <strong>{vel_trend}</strong>.</li>
                    <li>Changes in money velocity indicate shifts in the efficiency of money usage in the economy, which can reflect changing consumer behavior, financial innovation, or economic uncertainty.</li>
                    <li>The relationship between money supply growth and velocity changes suggests that monetary policy may need to adapt to maintain economic stability.</li>
                </ul>
                <p>
                    <strong>Policy Implications:</strong> Understanding these relationships is crucial for effective monetary policy formulation. Central banks need to consider not only changes in money supply but also changes in velocity when making policy decisions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.info("Insufficient December data for annual analysis. Please expand your year range selection.")

    # Download options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Download Data")
    
    # Convert dataframe to CSV for download
    csv = filtered_df.to_csv(index=False)
    
    st.sidebar.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="money_supply_data.csv",
        mime="text/csv",
    )

    # Add information about the dashboard
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this Dashboard**
    
    This dashboard visualizes the relationship between money supply metrics and their velocities over time, 
    helping to understand their impact on the economy.
    
    The data covers the period from 2003 to 2024, focusing on monetary aggregates and their economic implications.
    """)
