"""
SmartEnergySense: AI-Powered Household Energy Optimization
Main Streamlit Application - Week 1: Setup & Data Simulation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os

# Import our modules
from utils.helpers import (
    validate_energy_data, process_uploaded_data, calculate_daily_stats,
    calculate_hourly_patterns, detect_anomalies, get_energy_insights,
    create_sample_data_if_needed, load_data, format_energy_value,
    analyze_energy_by_device, calculate_energy_savings
)

from utils.auth import (
    register_user, login_user, logout_user, is_logged_in, get_user_profile
)

# Page configuration
st.set_page_config(
    page_title="SmartEnergySense",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">⚡ SmartEnergySense</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Household Energy Optimization")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Management")
        
        # Data source selection
        data_source = st.radio(
            "Choose your data source:",
            ["📁 Upload CSV", "🎲 Use Sample Data", "📊 Demo Mode"]
        )
        
        if data_source == "📁 Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload your energy usage CSV file",
                type=['csv'],
                help="File should contain timestamp and energy data columns"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    is_valid, error_msg = validate_energy_data(df)
                    
                    if is_valid:
                        df = process_uploaded_data(df)
                        st.success("✅ Data loaded successfully!")
                        
                        # Show data info
                        with st.expander("📊 Data Information"):
                            st.write(f"**Rows:** {len(df)}")
                            st.write(f"**Date Range:** {df['timestamp'].min()} to {df['timestamp'].max()}")
                            st.write(f"**Energy Range:** {df['energy_kwh'].min():.2f} to {df['energy_kwh'].max():.2f} kWh")
                            st.write(f"**Average:** {df['energy_kwh'].mean():.2f} kWh")
                        
                        st.session_state['energy_data'] = df
                    else:
                        st.error(f"❌ {error_msg}")
                        
                        # Show available columns for debugging
                        st.write("**Available columns in your file:**")
                        st.write(list(df.columns))
                        st.write("**First few rows:**")
                        st.dataframe(df.head())
                        return
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    return
            else:
                st.info("Please upload a CSV file to get started.")
                return
                
        elif data_source == "🎲 Use Sample Data":
            if st.button("Generate Sample Data"):
                with st.spinner("Generating sample data..."):
                    sample_file = create_sample_data_if_needed()
                    df = load_data(sample_file)
                    if df is not None:
                        st.session_state['energy_data'] = df
                        st.success("✅ Sample data loaded!")
                    else:
                        st.error("❌ Failed to load sample data")
                        return
            else:
                st.info("Click the button to generate sample data.")
                return
                
        else:  # Demo Mode
            if st.button("Load Demo Data"):
                with st.spinner("Loading demo data..."):
                    # Create demo data on the fly
                    from src.data_generator import EnergyDataGenerator
                    generator = EnergyDataGenerator(days=30)
                    df = generator.generate_sample_data()
                    st.session_state['energy_data'] = df
                    st.success("✅ Demo data loaded!")
            else:
                st.info("Click the button to load demo data.")
                return
    
    # Main content
    if 'energy_data' not in st.session_state:
        st.info("👈 Please select a data source from the sidebar to get started.")
        return
    
    df = st.session_state['energy_data']
    
    # Dashboard
    st.header("📈 Energy Usage Dashboard")
    
    # Key metrics
    insights = get_energy_insights(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Energy Used",
            f"{format_energy_value(insights['total_energy'])}",
            help="Total energy consumption over the entire period"
        )
    
    with col2:
        st.metric(
            "Average Daily Usage",
            f"{format_energy_value(insights['avg_daily_energy'])}",
            help="Average daily energy consumption"
        )
    
    with col3:
        st.metric(
            "Peak Hour",
            f"{insights['peak_hour']}:00",
            help="Hour of the day with highest average usage"
        )
    
    with col4:
        weekend_diff = insights['weekend_avg'] - insights['weekday_avg']
        st.metric(
            "Weekend vs Weekday",
            f"{format_energy_value(weekend_diff)}",
            delta=f"{'↑' if weekend_diff > 0 else '↓'} {abs(weekend_diff/insights['weekday_avg']*100):.1f}%",
            help="Difference between weekend and weekday usage"
        )
    
    # Charts
    st.subheader("📊 Energy Usage Patterns")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Daily Trends", "🕐 Hourly Patterns", "📈 Time Series", "🔍 Anomalies", "💡 Energy Analysis"])
    
    with tab1:
        st.subheader("Daily Energy Consumption")
        
        daily_stats = calculate_daily_stats(df)
        
        # Daily consumption chart
        fig_daily = px.line(
            daily_stats,
            x='date',
            y='total_kwh',
            title="Daily Energy Consumption",
            labels={'total_kwh': 'Energy (kWh)', 'date': 'Date'}
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Daily statistics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Daily Statistics")
            st.dataframe(
                daily_stats.round(2),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.subheader("📊 Summary Statistics")
            summary_stats = {
                'Average Daily Usage': f"{daily_stats['total_kwh'].mean():.2f} kWh",
                'Highest Daily Usage': f"{daily_stats['total_kwh'].max():.2f} kWh",
                'Lowest Daily Usage': f"{daily_stats['total_kwh'].min():.2f} kWh",
                'Standard Deviation': f"{daily_stats['total_kwh'].std():.2f} kWh",
                'Total Days': f"{len(daily_stats)} days"
            }
            
            for stat, value in summary_stats.items():
                st.metric(stat, value)
    
    with tab2:
        st.subheader("Hourly Energy Patterns")
        
        hourly_patterns = calculate_hourly_patterns(df)
        
        # Hourly pattern chart
        fig_hourly = px.bar(
            hourly_patterns,
            x='hour',
            y='avg_kwh',
            title="Average Hourly Energy Usage",
            labels={'avg_kwh': 'Average Energy (kWh)', 'hour': 'Hour of Day'},
            error_y='std_kwh'
        )
        fig_hourly.update_layout(height=400)
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Peak hours analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🕐 Peak Hours Analysis")
            peak_hour = hourly_patterns.loc[hourly_patterns['avg_kwh'].idxmax()]
            st.metric(
                "Peak Hour",
                f"{peak_hour['hour']}:00",
                f"{peak_hour['avg_kwh']:.2f} kWh"
            )
            
            # Top 3 peak hours
            top_3_hours = hourly_patterns.nlargest(3, 'avg_kwh')
            st.write("**Top 3 Peak Hours:**")
            for _, row in top_3_hours.iterrows():
                st.write(f"• {row['hour']}:00 - {row['avg_kwh']:.2f} kWh")
        
        with col2:
            st.subheader("📊 Hourly Statistics")
            st.dataframe(
                hourly_patterns.round(3),
                use_container_width=True,
                hide_index=True
            )
    
    with tab3:
        st.subheader("Time Series Analysis")
        
        # Time series chart
        fig_ts = px.line(
            df,
            x='timestamp',
            y='energy_kwh',
            title="Energy Usage Over Time",
            labels={'energy_kwh': 'Energy (kWh)', 'timestamp': 'Time'}
        )
        fig_ts.update_layout(height=400)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Rolling average
        df_rolling = df.copy()
        df_rolling['rolling_avg'] = df_rolling['energy_kwh'].rolling(window=24).mean()
        
        fig_rolling = px.line(
            df_rolling,
            x='timestamp',
            y=['energy_kwh', 'rolling_avg'],
            title="Energy Usage with 24-Hour Rolling Average",
            labels={'value': 'Energy (kWh)', 'timestamp': 'Time', 'variable': 'Metric'}
        )
        fig_rolling.update_layout(height=400)
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    with tab4:
        st.subheader("Anomaly Detection")
        
        # Detect anomalies
        anomalies = detect_anomalies(df, threshold=2.0)
        
        if len(anomalies) > 0:
            st.warning(f"⚠️ Found {len(anomalies)} anomalies in your data!")
            
            # Anomaly chart
            fig_anomaly = px.scatter(
                df,
                x='timestamp',
                y='energy_kwh',
                title="Energy Usage with Anomalies Highlighted",
                labels={'energy_kwh': 'Energy (kWh)', 'timestamp': 'Time'}
            )
            
            # Add anomaly points
            fig_anomaly.add_scatter(
                x=anomalies['timestamp'],
                y=anomalies['energy_kwh'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Anomalies'
            )
            
            fig_anomaly.update_layout(height=400)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly details
            st.subheader("🔍 Anomaly Details")
            anomaly_details = anomalies[['timestamp', 'energy_kwh', 'z_score']].copy()
            anomaly_details['z_score'] = anomaly_details['z_score'].round(2)
            st.dataframe(anomaly_details, use_container_width=True)
        else:
            st.success("✅ No significant anomalies detected in your data!")
    
    with tab5:
        st.subheader("💡 Energy Analysis by Device")
        
        # Analyze energy consumption by device
        device_analysis = analyze_energy_by_device(df)
        savings_analysis = calculate_energy_savings(df)
        
        # Display total energy and potential savings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Energy Used",
                f"{format_energy_value(savings_analysis['total_energy'])}",
                help="Total energy consumption over the entire period"
            )
        
        with col2:
            st.metric(
                "Potential Savings",
                f"{format_energy_value(savings_analysis['total_potential_savings'])}",
                delta=f"↓ {savings_analysis['savings_percentage']:.1f}%",
                help="Potential energy savings with optimizations"
            )
        
        with col3:
            st.metric(
                "Savings Percentage",
                f"{savings_analysis['savings_percentage']:.1f}%",
                help="Percentage of total energy that could be saved"
            )
        
        # Device consumption breakdown
        st.subheader("📊 Device Energy Consumption Breakdown")
        
        # Create device consumption chart
        device_data = []
        for device, analysis in device_analysis.items():
            if analysis['estimated_consumption'] > 0:
                device_data.append({
                    'Device': device,
                    'Consumption (kWh)': analysis['estimated_consumption'],
                    'Percentage': analysis['percentage_of_total'],
                    'Potential Savings (kWh)': analysis['potential_savings']
                })
        
        device_df = pd.DataFrame(device_data)
        
        if len(device_df) > 0:
            # Device consumption pie chart
            fig_device = px.pie(
                device_df,
                values='Consumption (kWh)',
                names='Device',
                title="Energy Consumption by Device",
                hover_data=['Percentage']
            )
            fig_device.update_layout(height=400)
            st.plotly_chart(fig_device, use_container_width=True)
            
            # Device analysis table
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Device Consumption Details")
                display_df = device_df[['Device', 'Consumption (kWh)', 'Percentage']].copy()
                display_df['Percentage'] = display_df['Percentage'].round(1)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("💰 Potential Savings by Device")
                savings_df = device_df[['Device', 'Potential Savings (kWh)']].copy()
                savings_df = savings_df.sort_values('Potential Savings (kWh)', ascending=False)
                st.dataframe(savings_df, use_container_width=True, hide_index=True)
        
        # Energy savings recommendations
        st.subheader("🎯 Energy Savings Recommendations")
        
        for recommendation in savings_analysis['recommendations']:
            with st.expander(f"💡 {recommendation['device']} - {format_energy_value(recommendation['potential_savings'])} potential savings"):
                st.write(f"**Current Consumption:** {format_energy_value(recommendation['current_consumption'])}")
                st.write(f"**Potential Savings:** {format_energy_value(recommendation['potential_savings'])} ({recommendation['percentage_of_total']:.1f}% of total)")
                st.write("**Recommendations:**")
                for rec in recommendation['recommendation']:
                    st.write(f"• {rec}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>SmartEnergySense - AI-Powered Household Energy Optimization</p>
            <p>@Branch (Diff with Main Branch) @mrecw</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def splash_screen():
    st.markdown(
        """
        <style>
            .splash-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                background: linear-gradient(135deg, #1f77b4, #6baed6);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .splash-title {
                font-size: 4rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .splash-subtitle {
                font-size: 1.5rem;
                margin-bottom: 1rem;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
            }
            .splash-loading {
                font-size: 1.2rem;
                color: #d0d0d0;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            }
        </style>
        <div class="splash-container">
            <h1 class="splash-title">⚡ Welcome to SmartEnergySense</h1>
            <p class="splash-subtitle">AI-Powered Household Energy Optimization</p>
            <p class="splash-loading">Loading...</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def login_page():
    st.markdown(
        """
        <style>
            .login-container {
                max-width: 400px;
                margin: 3rem auto;
                padding: 2rem;
                background-color: #f0f2f6;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .login-title {
                text-align: center;
                color: #1f77b4;
                font-weight: 700;
                margin-bottom: 1.5rem;
                font-size: 2.5rem;
            }
            .login-button {
                background-color: #1f77b4;
                color: white;
                width: 100%;
                padding: 0.75rem;
                border: none;
                border-radius: 5px;
                font-size: 1rem;
                cursor: pointer;
                margin-top: 1rem;
            }
            .login-button:hover {
                background-color: #155d8b;
            }
            .link-button {
                background: none;
                border: none;
                color: #1f77b4;
                cursor: pointer;
                text-decoration: underline;
                font-size: 0.9rem;
                margin-top: 1rem;
                display: block;
                text-align: center;
            }
        </style>
        <div class="login-container">
            <h2 class="login-title">Login</h2>
    """,
        unsafe_allow_html=True
    )
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        success, msg = login_user(username, password)
        if success:
            st.success(msg)
            st.session_state['page'] = 'main_app'
            st.experimental_rerun()
        else:
            st.error(msg)
    if st.button("Go to Register", key="go_register_button"):
        st.session_state['page'] = 'register'
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def register_page():
    st.markdown(
        """
        <style>
            .register-container {
                max-width: 400px;
                margin: 3rem auto;
                padding: 2rem;
                background-color: #f0f2f6;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .register-title {
                text-align: center;
                color: #1f77b4;
                font-weight: 700;
                margin-bottom: 1.5rem;
                font-size: 2.5rem;
            }
            .register-button {
                background-color: #1f77b4;
                color: white;
                width: 100%;
                padding: 0.75rem;
                border: none;
                border-radius: 5px;
                font-size: 1rem;
                cursor: pointer;
                margin-top: 1rem;
            }
            .register-button:hover {
                background-color: #155d8b;
            }
            .link-button {
                background: none;
                border: none;
                color: #1f77b4;
                cursor: pointer;
                text-decoration: underline;
                font-size: 0.9rem;
                margin-top: 1rem;
                display: block;
                text-align: center;
            }
        </style>
        <div class="register-container">
            <h2 class="register-title">Register</h2>
    """,
        unsafe_allow_html=True
    )
    username = st.text_input("Choose a Username", key="register_username")
    password = st.text_input("Choose a Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
    if st.button("Register", key="register_button"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        else:
            success, msg = register_user(username, password)
            if success:
                st.success(msg)
                st.session_state['page'] = 'login'
                st.experimental_rerun()
            else:
                st.error(msg)
    if st.button("Go to Login", key="go_login_button"):
        st.session_state['page'] = 'login'
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def register_page():
    st.title("Register")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        else:
            success, msg = register_user(username, password)
            if success:
                st.success(msg)
                st.session_state['page'] = 'login'
                st.experimental_rerun()
            else:
                st.error(msg)
    if st.button("Go to Login"):
        st.session_state['page'] = 'login'
        st.experimental_rerun()

def main_app():
    # Header
    st.markdown('<h1 class="main-header">⚡ SmartEnergySense</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Household Energy Optimization")
    
    # Logout button
    if st.sidebar.button("Logout"):
        logout_user()
        st.session_state['page'] = 'login'
        st.experimental_rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Management")
        
        # Data source selection
        data_source = st.radio(
            "Choose your data source:",
            ["📁 Upload CSV", "🎲 Use Sample Data", "📊 Demo Mode"]
        )
        
        if data_source == "📁 Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload your energy usage CSV file",
                type=['csv'],
                help="File should contain timestamp and energy data columns"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    is_valid, error_msg = validate_energy_data(df)
                    
                    if is_valid:
                        df = process_uploaded_data(df)
                        st.success("✅ Data loaded successfully!")
                        
                        # Show data info
                        with st.expander("📊 Data Information"):
                            st.write(f"**Rows:** {len(df)}")
                            st.write(f"**Date Range:** {df['timestamp'].min()} to {df['timestamp'].max()}")
                            st.write(f"**Energy Range:** {df['energy_kwh'].min():.2f} to {df['energy_kwh'].max():.2f} kWh")
                            st.write(f"**Average:** {df['energy_kwh'].mean():.2f} kWh")
                        
                        st.session_state['energy_data'] = df
                    else:
                        st.error(f"❌ {error_msg}")
                        
                        # Show available columns for debugging
                        st.write("**Available columns in your file:**")
                        st.write(list(df.columns))
                        st.write("**First few rows:**")
                        st.dataframe(df.head())
                        return
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    return
            else:
                st.info("Please upload a CSV file to get started.")
                return
                
        elif data_source == "🎲 Use Sample Data":
            if st.button("Generate Sample Data"):
                with st.spinner("Generating sample data..."):
                    sample_file = create_sample_data_if_needed()
                    df = load_data(sample_file)
                    if df is not None:
                        st.session_state['energy_data'] = df
                        st.success("✅ Sample data loaded!")
                    else:
                        st.error("❌ Failed to load sample data")
                        return
            else:
                st.info("Click the button to generate sample data.")
                return
                
        else:  # Demo Mode
            if st.button("Load Demo Data"):
                with st.spinner("Loading demo data..."):
                    # Create demo data on the fly
                    from src.data_generator import EnergyDataGenerator
                    generator = EnergyDataGenerator(days=30)
                    df = generator.generate_sample_data()
                    st.session_state['energy_data'] = df
                    st.success("✅ Demo data loaded!")
            else:
                st.info("Click the button to load demo data.")
                return
    
    # Main content
    if 'energy_data' not in st.session_state:
        st.info("👈 Please select a data source from the sidebar to get started.")
        return
    
    df = st.session_state['energy_data']
    
    # Dashboard
    st.header("📈 Energy Usage Dashboard")
    
    # Key metrics
    insights = get_energy_insights(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Energy Used",
            f"{format_energy_value(insights['total_energy'])}",
            help="Total energy consumption over the entire period"
        )
    
    with col2:
        st.metric(
            "Average Daily Usage",
            f"{format_energy_value(insights['avg_daily_energy'])}",
            help="Average daily energy consumption"
        )
    
    with col3:
        st.metric(
            "Peak Hour",
            f"{insights['peak_hour']}:00",
            help="Hour of the day with highest average usage"
        )
    
    with col4:
        weekend_diff = insights['weekend_avg'] - insights['weekday_avg']
        st.metric(
            "Weekend vs Weekday",
            f"{format_energy_value(weekend_diff)}",
            delta=f"{'↑' if weekend_diff > 0 else '↓'} {abs(weekend_diff/insights['weekday_avg']*100):.1f}%",
            help="Difference between weekend and weekday usage"
        )
    
    # Charts
    st.subheader("📊 Energy Usage Patterns")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Daily Trends", "🕐 Hourly Patterns", "📈 Time Series", "🔍 Anomalies", "💡 Energy Analysis"])
    
    with tab1:
        st.subheader("Daily Energy Consumption")
        
        daily_stats = calculate_daily_stats(df)
        
        # Daily consumption chart
        fig_daily = px.line(
            daily_stats,
            x='date',
            y='total_kwh',
            title="Daily Energy Consumption",
            labels={'total_kwh': 'Energy (kWh)', 'date': 'Date'}
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Daily statistics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Daily Statistics")
            st.dataframe(
                daily_stats.round(2),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.subheader("📊 Summary Statistics")
            summary_stats = {
                'Average Daily Usage': f"{daily_stats['total_kwh'].mean():.2f} kWh",
                'Highest Daily Usage': f"{daily_stats['total_kwh'].max():.2f} kWh",
                'Lowest Daily Usage': f"{daily_stats['total_kwh'].min():.2f} kWh",
                'Standard Deviation': f"{daily_stats['total_kwh'].std():.2f} kWh",
                'Total Days': f"{len(daily_stats)} days"
            }
            
            for stat, value in summary_stats.items():
                st.metric(stat, value)
    
    with tab2:
        st.subheader("Hourly Energy Patterns")
        
        hourly_patterns = calculate_hourly_patterns(df)
        
        # Hourly pattern chart
        fig_hourly = px.bar(
            hourly_patterns,
            x='hour',
            y='avg_kwh',
            title="Average Hourly Energy Usage",
            labels={'avg_kwh': 'Average Energy (kWh)', 'hour': 'Hour of Day'},
            error_y='std_kwh'
        )
        fig_hourly.update_layout(height=400)
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Peak hours analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🕐 Peak Hours Analysis")
            peak_hour = hourly_patterns.loc[hourly_patterns['avg_kwh'].idxmax()]
            st.metric(
                "Peak Hour",
                f"{peak_hour['hour']}:00",
                f"{peak_hour['avg_kwh']:.2f} kWh"
            )
            
            # Top 3 peak hours
            top_3_hours = hourly_patterns.nlargest(3, 'avg_kwh')
            st.write("**Top 3 Peak Hours:**")
            for _, row in top_3_hours.iterrows():
                st.write(f"• {row['hour']}:00 - {row['avg_kwh']:.2f} kWh")
        
        with col2:
            st.subheader("📊 Hourly Statistics")
            st.dataframe(
                hourly_patterns.round(3),
                use_container_width=True,
                hide_index=True
            )
    
    with tab3:
        st.subheader("Time Series Analysis")
        
        # Time series chart
        fig_ts = px.line(
            df,
            x='timestamp',
            y='energy_kwh',
            title="Energy Usage Over Time",
            labels={'energy_kwh': 'Energy (kWh)', 'timestamp': 'Time'}
        )
        fig_ts.update_layout(height=400)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Rolling average
        df_rolling = df.copy()
        df_rolling['rolling_avg'] = df_rolling['energy_kwh'].rolling(window=24).mean()
        
        fig_rolling = px.line(
            df_rolling,
            x='timestamp',
            y=['energy_kwh', 'rolling_avg'],
            title="Energy Usage with 24-Hour Rolling Average",
            labels={'value': 'Energy (kWh)', 'timestamp': 'Time', 'variable': 'Metric'}
        )
        fig_rolling.update_layout(height=400)
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    with tab4:
        st.subheader("Anomaly Detection")
        
        # Detect anomalies
        anomalies = detect_anomalies(df, threshold=2.0)
        
        if len(anomalies) > 0:
            st.warning(f"⚠️ Found {len(anomalies)} anomalies in your data!")
            
            # Anomaly chart
            fig_anomaly = px.scatter(
                df,
                x='timestamp',
                y='energy_kwh',
                title="Energy Usage with Anomalies Highlighted",
                labels={'energy_kwh': 'Energy (kWh)', 'timestamp': 'Time'}
            )
            
            # Add anomaly points
            fig_anomaly.add_scatter(
                x=anomalies['timestamp'],
                y=anomalies['energy_kwh'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Anomalies'
            )
            
            fig_anomaly.update_layout(height=400)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly details
            st.subheader("🔍 Anomaly Details")
            anomaly_details = anomalies[['timestamp', 'energy_kwh', 'z_score']].copy()
            anomaly_details['z_score'] = anomaly_details['z_score'].round(2)
            st.dataframe(anomaly_details, use_container_width=True)
        else:
            st.success("✅ No significant anomalies detected in your data!")
    
    with tab5:
        st.subheader("💡 Energy Analysis by Device")
        
        # Analyze energy consumption by device
        device_analysis = analyze_energy_by_device(df)
        savings_analysis = calculate_energy_savings(df)
        
        # Display total energy and potential savings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Energy Used",
                f"{format_energy_value(savings_analysis['total_energy'])}",
                help="Total energy consumption over the entire period"
            )
        
        with col2:
            st.metric(
                "Potential Savings",
                f"{format_energy_value(savings_analysis['total_potential_savings'])}",
                delta=f"↓ {savings_analysis['savings_percentage']:.1f}%",
                help="Potential energy savings with optimizations"
            )
        
        with col3:
            st.metric(
                "Savings Percentage",
                f"{savings_analysis['savings_percentage']:.1f}%",
                help="Percentage of total energy that could be saved"
            )
        
        # Device consumption breakdown
        st.subheader("📊 Device Energy Consumption Breakdown")
        
        # Create device consumption chart
        device_data = []
        for device, analysis in device_analysis.items():
            if analysis['estimated_consumption'] > 0:
                device_data.append({
                    'Device': device,
                    'Consumption (kWh)': analysis['estimated_consumption'],
                    'Percentage': analysis['percentage_of_total'],
                    'Potential Savings (kWh)': analysis['potential_savings']
                })
        
        device_df = pd.DataFrame(device_data)
        
        if len(device_df) > 0:
            # Device consumption pie chart
            fig_device = px.pie(
                device_df,
                values='Consumption (kWh)',
                names='Device',
                title="Energy Consumption by Device",
                hover_data=['Percentage']
            )
            fig_device.update_layout(height=400)
            st.plotly_chart(fig_device, use_container_width=True)
            
            # Device analysis table
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Device Consumption Details")
                display_df = device_df[['Device', 'Consumption (kWh)', 'Percentage']].copy()
                display_df['Percentage'] = display_df['Percentage'].round(1)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("💰 Potential Savings by Device")
                savings_df = device_df[['Device', 'Potential Savings (kWh)']].copy()
                savings_df = savings_df.sort_values('Potential Savings (kWh)', ascending=False)
                st.dataframe(savings_df, use_container_width=True, hide_index=True)
        
        # Energy savings recommendations
        st.subheader("🎯 Energy Savings Recommendations")
        
        for recommendation in savings_analysis['recommendations']:
            with st.expander(f"💡 {recommendation['device']} - {format_energy_value(recommendation['potential_savings'])} potential savings"):
                st.write(f"**Current Consumption:** {format_energy_value(recommendation['current_consumption'])}")
                st.write(f"**Potential Savings:** {format_energy_value(recommendation['potential_savings'])} ({recommendation['percentage_of_total']:.1f}% of total)")
                st.write("**Recommendations:**")
                for rec in recommendation['recommendation']:
                    st.write(f"• {rec}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>SmartEnergySense - AI-Powered Household Energy Optimization</p>
            <p>@Branch (Diff with Main Branch) @mrecw</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def profile_page():
    st.markdown("## User Profile")
    username = st.session_state.get('username', None)
    if not username:
        st.error("User not logged in.")
        return
    profile = get_user_profile(username)
    name = st.text_input("Name", value=profile.get("name", ""))
    email = st.text_input("Email", value=profile.get("email", ""))
    preferences = st.text_area("Preferences (JSON format)", value=str(profile.get("preferences", {})))
    if st.button("Update Profile"):
        try:
            import json
            prefs = json.loads(preferences)
            new_profile = {
                "name": name,
                "email": email,
                "preferences": prefs
            }
            success, msg = update_user_profile(username, new_profile)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        except Exception as e:
            st.error(f"Invalid preferences JSON: {e}")

def generate_report():
    import io
    import pandas as pd
    from datetime import datetime

    if 'energy_data' not in st.session_state:
        st.error("No energy data available to generate report.")
        return None

    df = st.session_state['energy_data']
    insights = get_energy_insights(df)
    anomalies = detect_anomalies(df, threshold=2.0)
    savings_analysis = calculate_energy_savings(df)

    output = io.StringIO()
    output.write("SmartEnergySense Energy Report\n")
    output.write(f"Generated on: {datetime.now()}\n\n")

    output.write("Energy Usage Summary:\n")
    for key, value in insights.items():
        output.write(f"{key}: {value}\n")
    output.write("\n")

    output.write("Anomalies Detected:\n")
    if len(anomalies) > 0:
        anomalies.to_csv(output, index=False)
    else:
        output.write("No significant anomalies detected.\n")
    output.write("\n")

    output.write("Energy Savings Recommendations:\n")
    for rec in savings_analysis['recommendations']:
        output.write(f"- {rec['device']}: {rec['potential_savings']} kWh potential savings\n")
        for r in rec['recommendation']:
            output.write(f"  * {r}\n")
    output.seek(0)
    return output

def main():
    import time
    if 'page' not in st.session_state:
        st.session_state['page'] = 'splash'
        st.session_state['splash_start'] = time.time()
    
    if st.session_state['page'] == 'splash':
        splash_screen()
        # Auto transition after 5 seconds
        if time.time() - st.session_state['splash_start'] > 5:
            st.session_state['page'] = 'login'
            st.experimental_rerun()
        if st.button("Continue to Login"):
            st.session_state['page'] = 'login'
            st.experimental_rerun()
    elif st.session_state['page'] == 'login':
        login_page()
    elif st.session_state['page'] == 'register':
        register_page()
    elif st.session_state['page'] == 'profile':
        if is_logged_in():
            profile_page()
        else:
            st.session_state['page'] = 'login'
            st.experimental_rerun()
    elif st.session_state['page'] == 'main_app':
        if is_logged_in():
            main_app()
        else:
            st.session_state['page'] = 'login'
            st.experimental_rerun()

def main_app():
    # Header
    st.markdown('<h1 class="main-header">⚡ SmartEnergySense</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Household Energy Optimization")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Logout"):
            logout_user()
            st.session_state['page'] = 'login'
            st.experimental_rerun()
    with col2:
        if st.button("Profile"):
            st.session_state['page'] = 'profile'
            st.experimental_rerun()
    with col3:
        if st.button("Download Report"):
            report = generate_report()
            if report:
                st.download_button(
                    label="Download Energy Report",
                    data=report.getvalue(),
                    file_name="energy_report.txt",
                    mime="text/plain"
                )
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Management")
        
        # Data source selection
        data_source = st.radio(
            "Choose your data source:",
            ["📁 Upload CSV", "🎲 Use Sample Data", "📊 Demo Mode"]
        )
        
        if data_source == "📁 Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload your energy usage CSV file",
                type=['csv'],
                help="File should contain timestamp and energy data columns"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    is_valid, error_msg = validate_energy_data(df)
                    
                    if is_valid:
                        df = process_uploaded_data(df)
                        st.success("✅ Data loaded successfully!")
                        
                        # Show data info
                        with st.expander("📊 Data Information"):
                            st.write(f"**Rows:** {len(df)}")
                            st.write(f"**Date Range:** {df['timestamp'].min()} to {df['timestamp'].max()}")
                            st.write(f"**Energy Range:** {df['energy_kwh'].min():.2f} to {df['energy_kwh'].max():.2f} kWh")
                            st.write(f"**Average:** {df['energy_kwh'].mean():.2f} kWh")
                        
                        st.session_state['energy_data'] = df
                    else:
                        st.error(f"❌ {error_msg}")
                        
                        # Show available columns for debugging
                        st.write("**Available columns in your file:**")
                        st.write(list(df.columns))
                        st.write("**First few rows:**")
                        st.dataframe(df.head())
                        return
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    return
            else:
                st.info("Please upload a CSV file to get started.")
                return
                
        elif data_source == "🎲 Use Sample Data":
            if st.button("Generate Sample Data"):
                with st.spinner("Generating sample data..."):
                    sample_file = create_sample_data_if_needed()
                    df = load_data(sample_file)
                    if df is not None:
                        st.session_state['energy_data'] = df
                        st.success("✅ Sample data loaded!")
                    else:
                        st.error("❌ Failed to load sample data")
                        return
            else:
                st.info("Click the button to generate sample data.")
                return
                
        else:  # Demo Mode
            if st.button("Load Demo Data"):
                with st.spinner("Loading demo data..."):
                    # Create demo data on the fly
                    from src.data_generator import EnergyDataGenerator
                    generator = EnergyDataGenerator(days=30)
                    df = generator.generate_sample_data()
                    st.session_state['energy_data'] = df
                    st.success("✅ Demo data loaded!")
            else:
                st.info("Click the button to load demo data.")
                return
    
    # Main content
    if 'energy_data' not in st.session_state:
        st.info("👈 Please select a data source from the sidebar to get started.")
        return
    
    df = st.session_state['energy_data']
    
    # Dashboard
    st.header("📈 Energy Usage Dashboard")
    
    # Key metrics
    insights = get_energy_insights(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Energy Used",
            f"{format_energy_value(insights['total_energy'])}",
            help="Total energy consumption over the entire period"
        )
    
    with col2:
        st.metric(
            "Average Daily Usage",
            f"{format_energy_value(insights['avg_daily_energy'])}",
            help="Average daily energy consumption"
        )
    
    with col3:
        st.metric(
            "Peak Hour",
            f"{insights['peak_hour']}:00",
            help="Hour of the day with highest average usage"
        )
    
    with col4:
        weekend_diff = insights['weekend_avg'] - insights['weekday_avg']
        st.metric(
            "Weekend vs Weekday",
            f"{format_energy_value(weekend_diff)}",
            delta=f"{'↑' if weekend_diff > 0 else '↓'} {abs(weekend_diff/insights['weekday_avg']*100):.1f}%",
            help="Difference between weekend and weekday usage"
        )
    
    # Charts
    st.subheader("📊 Energy Usage Patterns")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Daily Trends", "🕐 Hourly Patterns", "📈 Time Series", "🔍 Anomalies", "💡 Energy Analysis"])
    
    with tab1:
        st.subheader("Daily Energy Consumption")
        
        daily_stats = calculate_daily_stats(df)
        
        # Daily consumption chart
        fig_daily = px.line(
            daily_stats,
            x='date',
            y='total_kwh',
            title="Daily Energy Consumption",
            labels={'total_kwh': 'Energy (kWh)', 'date': 'Date'}
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Daily statistics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Daily Statistics")
            st.dataframe(
                daily_stats.round(2),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.subheader("📊 Summary Statistics")
            summary_stats = {
                'Average Daily Usage': f"{daily_stats['total_kwh'].mean():.2f} kWh",
                'Highest Daily Usage': f"{daily_stats['total_kwh'].max():.2f} kWh",
                'Lowest Daily Usage': f"{daily_stats['total_kwh'].min():.2f} kWh",
                'Standard Deviation': f"{daily_stats['total_kwh'].std():.2f} kWh",
                'Total Days': f"{len(daily_stats)} days"
            }
            
            for stat, value in summary_stats.items():
                st.metric(stat, value)
    
    with tab2:
        st.subheader("Hourly Energy Patterns")
        
        hourly_patterns = calculate_hourly_patterns(df)
        
        # Hourly pattern chart
        fig_hourly = px.bar(
            hourly_patterns,
            x='hour',
            y='avg_kwh',
            title="Average Hourly Energy Usage",
            labels={'avg_kwh': 'Average Energy (kWh)', 'hour': 'Hour of Day'},
            error_y='std_kwh'
        )
        fig_hourly.update_layout(height=400)
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Peak hours analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🕐 Peak Hours Analysis")
            peak_hour = hourly_patterns.loc[hourly_patterns['avg_kwh'].idxmax()]
            st.metric(
                "Peak Hour",
                f"{peak_hour['hour']}:00",
                f"{peak_hour['avg_kwh']:.2f} kWh"
            )
            
            # Top 3 peak hours
            top_3_hours = hourly_patterns.nlargest(3, 'avg_kwh')
            st.write("**Top 3 Peak Hours:**")
            for _, row in top_3_hours.iterrows():
                st.write(f"• {row['hour']}:00 - {row['avg_kwh']:.2f} kWh")
        
        with col2:
            st.subheader("📊 Hourly Statistics")
            st.dataframe(
                hourly_patterns.round(3),
                use_container_width=True,
                hide_index=True
            )
    
    with tab3:
        st.subheader("Time Series Analysis")
        
        # Time series chart
        fig_ts = px.line(
            df,
            x='timestamp',
            y='energy_kwh',
            title="Energy Usage Over Time",
            labels={'energy_kwh': 'Energy (kWh)', 'timestamp': 'Time'}
        )
        fig_ts.update_layout(height=400)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Rolling average
        df_rolling = df.copy()
        df_rolling['rolling_avg'] = df_rolling['energy_kwh'].rolling(window=24).mean()
        
        fig_rolling = px.line(
            df_rolling,
            x='timestamp',
            y=['energy_kwh', 'rolling_avg'],
            title="Energy Usage with 24-Hour Rolling Average",
            labels={'value': 'Energy (kWh)', 'timestamp': 'Time', 'variable': 'Metric'}
        )
        fig_rolling.update_layout(height=400)
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    with tab4:
        st.subheader("Anomaly Detection")
        
        # Detect anomalies
        anomalies = detect_anomalies(df, threshold=2.0)
        
        if len(anomalies) > 0:
            st.warning(f"⚠️ Found {len(anomalies)} anomalies in your data!")
            
            # Anomaly chart
            fig_anomaly = px.scatter(
                df,
                x='timestamp',
                y='energy_kwh',
                title="Energy Usage with Anomalies Highlighted",
                labels={'energy_kwh': 'Energy (kWh)', 'timestamp': 'Time'}
            )
            
            # Add anomaly points
            fig_anomaly.add_scatter(
                x=anomalies['timestamp'],
                y=anomalies['energy_kwh'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Anomalies'
            )
            
            fig_anomaly.update_layout(height=400)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly details
            st.subheader("🔍 Anomaly Details")
            anomaly_details = anomalies[['timestamp', 'energy_kwh', 'z_score']].copy()
            anomaly_details['z_score'] = anomaly_details['z_score'].round(2)
            st.dataframe(anomaly_details, use_container_width=True)
        else:
            st.success("✅ No significant anomalies detected in your data!")
    
    with tab5:
        st.subheader("💡 Energy Analysis by Device")
        
        # Analyze energy consumption by device
        device_analysis = analyze_energy_by_device(df)
        savings_analysis = calculate_energy_savings(df)
        
        # Display total energy and potential savings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Energy Used",
                f"{format_energy_value(savings_analysis['total_energy'])}",
                help="Total energy consumption over the entire period"
            )
        
        with col2:
            st.metric(
                "Potential Savings",
                f"{format_energy_value(savings_analysis['total_potential_savings'])}",
                delta=f"↓ {savings_analysis['savings_percentage']:.1f}%",
                help="Potential energy savings with optimizations"
            )
        
        with col3:
            st.metric(
                "Savings Percentage",
                f"{savings_analysis['savings_percentage']:.1f}%",
                help="Percentage of total energy that could be saved"
            )
        
        # Device consumption breakdown
        st.subheader("📊 Device Energy Consumption Breakdown")
        
        # Create device consumption chart
        device_data = []
        for device, analysis in device_analysis.items():
            if analysis['estimated_consumption'] > 0:
                device_data.append({
                    'Device': device,
                    'Consumption (kWh)': analysis['estimated_consumption'],
                    'Percentage': analysis['percentage_of_total'],
                    'Potential Savings (kWh)': analysis['potential_savings']
                })
        
        device_df = pd.DataFrame(device_data)
        
        if len(device_df) > 0:
            # Device consumption pie chart
            fig_device = px.pie(
                device_df,
                values='Consumption (kWh)',
                names='Device',
                title="Energy Consumption by Device",
                hover_data=['Percentage']
            )
            fig_device.update_layout(height=400)
            st.plotly_chart(fig_device, use_container_width=True)
            
            # Device analysis table
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Device Consumption Details")
                display_df = device_df[['Device', 'Consumption (kWh)', 'Percentage']].copy()
                display_df['Percentage'] = display_df['Percentage'].round(1)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("💰 Potential Savings by Device")
                savings_df = device_df[['Device', 'Potential Savings (kWh)']].copy()
                savings_df = savings_df.sort_values('Potential Savings (kWh)', ascending=False)
                st.dataframe(savings_df, use_container_width=True, hide_index=True)
        
        # Energy savings recommendations
        st.subheader("🎯 Energy Savings Recommendations")
        
        for recommendation in savings_analysis['recommendations']:
            with st.expander(f"💡 {recommendation['device']} - {format_energy_value(recommendation['potential_savings'])} potential savings"):
                st.write(f"**Current Consumption:** {format_energy_value(recommendation['current_consumption'])}")
                st.write(f"**Potential Savings:** {format_energy_value(recommendation['potential_savings'])} ({recommendation['percentage_of_total']:.1f}% of total)")
                st.write("**Recommendations:**")
                for rec in recommendation['recommendation']:
                    st.write(f"• {rec}")
    

if __name__ == "__main__":
    main()
