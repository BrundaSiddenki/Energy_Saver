"""
Helper utilities for SmartEnergySense
Common functions for data processing, validation, and utilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import os
import re

def detect_csv_format(df):
    """
    Automatically detect the format of uploaded CSV and standardize it.
    
    Args:
        df (pd.DataFrame): Raw uploaded data
        
    Returns:
        pd.DataFrame: Standardized data with 'timestamp' and 'energy_kwh' columns
    """
    # Common energy-related column names
    energy_columns = [
        'energy_kwh', 'energy', 'kwh', 'power', 'consumption', 'usage',
        'electricity', 'watts', 'wattage', 'energy_consumption', 'load',
        'demand', 'current', 'voltage', 'amperage', 'reading', 'value',
        'measurement', 'data', 'usage_kwh', 'power_kw', 'energy_wh'
    ]
    
    time_columns = [
        'timestamp', 'time', 'date', 'datetime', 'created_at', 'recorded_at',
        'measurement_time', 'reading_time', 'period', 'interval', 'hour',
        'day', 'month', 'year', 'start_time', 'end_time'
    ]
    
    # Find energy column
    energy_col = None
    for col in df.columns:
        if any(energy_term in col.lower() for energy_term in energy_columns):
            energy_col = col
            break
    
    # If no energy column found, try to infer from data
    if energy_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Use the first numeric column as energy
            energy_col = numeric_cols[0]
    
    # Find timestamp column
    time_col = None
    for col in df.columns:
        if any(time_term in col.lower() for time_term in time_columns):
            time_col = col
            break
    
    # If no time column found, try to infer
    if time_col is None:
        # Look for datetime-like columns
        for col in df.columns:
            try:
                pd.to_datetime(df[col].iloc[0])
                time_col = col
                break
            except:
                continue
    
    # Create standardized dataframe
    if energy_col and time_col:
        # Convert energy values and detect units
        energy_values = pd.to_numeric(df[energy_col], errors='coerce')
        
        # Detect if values might be in different units and convert
        if energy_values.max() > 1000:
            # Values might be in watts or other units, but let's keep as is
            # User can specify if conversion is needed
            pass
        elif energy_values.max() < 0.1:
            # Values might be in kWh but very small, or in different units
            pass
        
        standardized_df = pd.DataFrame({
            'timestamp': pd.to_datetime(df[time_col]),
            'energy_kwh': energy_values
        })
        
        # Remove rows with invalid data
        standardized_df = standardized_df.dropna()
        
        # Add unit detection info
        unit_info = ""
        if len(standardized_df) > 0:
            max_val = standardized_df['energy_kwh'].max()
            if max_val > 1000:
                unit_info = f" (Note: High values detected - max {max_val:.2f})"
            elif max_val < 0.1:
                unit_info = f" (Note: Low values detected - max {max_val:.4f})"
        
        return standardized_df, True, f"Detected columns: {time_col} -> timestamp, {energy_col} -> energy_kwh{unit_info}"
    else:
        return None, False, f"Could not detect energy or time columns. Available columns: {list(df.columns)}"

def validate_energy_data(df):
    """
    Validate uploaded energy data format.
    
    Args:
        df (pd.DataFrame): Uploaded data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Try to standardize the data first
    standardized_df, success, message = detect_csv_format(df)
    
    if not success:
        return False, message
    
    # Check if we have data after standardization
    if len(standardized_df) == 0:
        return False, "No valid data found after processing"
    
    # Check for negative values
    if (standardized_df['energy_kwh'] < 0).any():
        return False, "Energy usage values cannot be negative"
    
    # Check for reasonable values (0-10000 kWh per hour - much more permissive)
    if (standardized_df['energy_kwh'] > 10000).any():
        return False, "Energy usage values seem unrealistic (>10000 kWh/hour)"
    
    # Check for extremely small values that might be in wrong units
    if (standardized_df['energy_kwh'] < 0.001).any():
        # This might be in watts instead of kWh, but let's allow it
        pass
    
    return True, f"Data validation passed. {message}"

def process_uploaded_data(df):
    """
    Process and clean uploaded energy data.
    
    Args:
        df (pd.DataFrame): Raw uploaded data
        
    Returns:
        pd.DataFrame: Cleaned and processed data
    """
    # Standardize the data first
    standardized_df, success, message = detect_csv_format(df)
    
    if not success:
        raise ValueError(message)
    
    # Sort by timestamp
    df_processed = standardized_df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove duplicates
    df_processed = df_processed.drop_duplicates(subset=['timestamp'])
    
    # Add useful columns
    df_processed['date'] = df_processed['timestamp'].dt.date
    df_processed['hour'] = df_processed['timestamp'].dt.hour
    df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
    df_processed['month'] = df_processed['timestamp'].dt.month
    df_processed['day_name'] = df_processed['timestamp'].dt.day_name()
    
    return df_processed

def analyze_energy_by_device(df):
    """
    Analyze energy consumption by different devices/appliances.
    
    Args:
        df (pd.DataFrame): Energy data
        
    Returns:
        dict: Device analysis results
    """
    # Define typical device energy consumption patterns
    device_patterns = {
        'Air Conditioner': {
            'peak_hours': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            'avg_consumption': 3.5,  # kWh per hour when running
            'typical_usage': 0.3  # 30% of total usage
        },
        'Refrigerator': {
            'peak_hours': list(range(24)),  # Runs continuously
            'avg_consumption': 0.15,  # kWh per hour
            'typical_usage': 0.15  # 15% of total usage
        },
        'Washing Machine': {
            'peak_hours': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'avg_consumption': 2.0,  # kWh per hour when running
            'typical_usage': 0.1  # 10% of total usage
        },
        'Dishwasher': {
            'peak_hours': [18, 19, 20, 21, 22],
            'avg_consumption': 1.8,  # kWh per hour when running
            'typical_usage': 0.08  # 8% of total usage
        },
        'Water Heater': {
            'peak_hours': [6, 7, 8, 9, 18, 19, 20, 21, 22],
            'avg_consumption': 4.5,  # kWh per hour when running
            'typical_usage': 0.18  # 18% of total usage
        },
        'Lighting': {
            'peak_hours': [6, 7, 8, 9, 17, 18, 19, 20, 21, 22, 23],
            'avg_consumption': 0.1,  # kWh per hour
            'typical_usage': 0.12  # 12% of total usage
        },
        'Electronics': {
            'peak_hours': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            'avg_consumption': 0.2,  # kWh per hour
            'typical_usage': 0.07  # 7% of total usage
        }
    }
    
    # Calculate total energy consumption
    total_energy = df['energy_kwh'].sum()
    avg_hourly_usage = df.groupby('hour')['energy_kwh'].mean()
    
    device_analysis = {}
    
    for device, pattern in device_patterns.items():
        # Calculate device-specific consumption based on patterns
        device_hours = pattern['peak_hours']
        device_consumption = avg_hourly_usage[avg_hourly_usage.index.isin(device_hours)].mean()
        
        # Estimate device usage
        estimated_usage = device_consumption * len(device_hours) * len(df) / 24
        potential_savings = estimated_usage * 0.2  # Assume 20% savings potential
        
        device_analysis[device] = {
            'estimated_consumption': estimated_usage,
            'percentage_of_total': (estimated_usage / total_energy) * 100,
            'potential_savings': potential_savings,
            'peak_hours': device_hours,
            'avg_consumption_per_hour': device_consumption
        }
    
    return device_analysis

def calculate_energy_savings(df):
    """
    Calculate potential energy savings recommendations.
    
    Args:
        df (pd.DataFrame): Energy data
        
    Returns:
        dict: Savings analysis
    """
    device_analysis = analyze_energy_by_device(df)
    total_energy = df['energy_kwh'].sum()
    
    # Calculate potential savings
    total_potential_savings = sum(device['potential_savings'] for device in device_analysis.values())
    
    # Generate savings recommendations
    savings_recommendations = []
    
    for device, analysis in device_analysis.items():
        if analysis['estimated_consumption'] > 0:
            savings_recommendations.append({
                'device': device,
                'current_consumption': analysis['estimated_consumption'],
                'potential_savings': analysis['potential_savings'],
                'percentage_of_total': analysis['percentage_of_total'],
                'recommendation': generate_device_recommendation(device, analysis)
            })
    
    # Sort by potential savings
    savings_recommendations.sort(key=lambda x: x['potential_savings'], reverse=True)
    
    return {
        'total_energy': total_energy,
        'total_potential_savings': total_potential_savings,
        'savings_percentage': (total_potential_savings / total_energy) * 100,
        'recommendations': savings_recommendations
    }

def generate_device_recommendation(device, analysis):
    """
    Generate specific recommendations for energy savings by device.
    
    Args:
        device (str): Device name
        analysis (dict): Device analysis data
        
    Returns:
        str: Recommendation text
    """
    recommendations = {
        'Air Conditioner': [
            "Set thermostat to 78°F (26°C) in summer",
            "Use ceiling fans to circulate air",
            "Clean or replace air filters monthly",
            "Consider a smart thermostat for better control"
        ],
        'Refrigerator': [
            "Keep refrigerator temperature at 37-40°F (3-4°C)",
            "Clean condenser coils regularly",
            "Don't leave door open unnecessarily",
            "Consider replacing if over 10 years old"
        ],
        'Washing Machine': [
            "Wash clothes in cold water when possible",
            "Run full loads only",
            "Use high-efficiency detergent",
            "Consider energy-efficient models"
        ],
        'Dishwasher': [
            "Run only when full",
            "Use energy-saving mode",
            "Skip the heated dry cycle",
            "Clean filter regularly"
        ],
        'Water Heater': [
            "Lower temperature to 120°F (49°C)",
            "Install a timer or smart controller",
            "Insulate hot water pipes",
            "Consider tankless water heater"
        ],
        'Lighting': [
            "Replace incandescent bulbs with LEDs",
            "Use motion sensors for outdoor lighting",
            "Turn off lights when leaving rooms",
            "Use natural light when possible"
        ],
        'Electronics': [
            "Unplug devices when not in use",
            "Use power strips with switches",
            "Enable sleep mode on computers",
            "Charge devices during off-peak hours"
        ]
    }
    
    if device in recommendations:
        return recommendations[device]
    else:
        return ["Check device efficiency", "Consider upgrading to energy-efficient model"]

def calculate_daily_stats(df):
    """
    Calculate daily energy usage statistics.
    
    Args:
        df (pd.DataFrame): Energy data
        
    Returns:
        pd.DataFrame: Daily statistics
    """
    daily_stats = df.groupby('date').agg({
        'energy_kwh': ['sum', 'mean', 'max', 'min', 'std']
    }).round(2)
    
    daily_stats.columns = ['total_kwh', 'avg_kwh', 'max_kwh', 'min_kwh', 'std_kwh']
    daily_stats = daily_stats.reset_index()
    
    return daily_stats

def calculate_hourly_patterns(df):
    """
    Calculate average hourly energy usage patterns.
    
    Args:
        df (pd.DataFrame): Energy data
        
    Returns:
        pd.DataFrame: Hourly patterns
    """
    hourly_patterns = df.groupby('hour')['energy_kwh'].agg(['mean', 'std']).round(3)
    hourly_patterns.columns = ['avg_kwh', 'std_kwh']
    hourly_patterns = hourly_patterns.reset_index()
    
    return hourly_patterns

def detect_anomalies(df, threshold=2.0):
    """
    Detect energy usage anomalies using statistical methods.
    
    Args:
        df (pd.DataFrame): Energy data
        threshold (float): Standard deviation threshold for anomaly detection
        
    Returns:
        pd.DataFrame: Anomaly data
    """
    # Calculate rolling statistics
    df['rolling_mean'] = df['energy_kwh'].rolling(window=24, center=True).mean()
    df['rolling_std'] = df['energy_kwh'].rolling(window=24, center=True).std()
    
    # Detect anomalies
    df['z_score'] = (df['energy_kwh'] - df['rolling_mean']) / df['rolling_std']
    df['is_anomaly'] = abs(df['z_score']) > threshold
    
    # Get anomaly data
    anomalies = df[df['is_anomaly']].copy()
    
    return anomalies

def format_energy_value(value, unit='kWh'):
    """
    Format energy values for display.
    
    Args:
        value (float): Energy value
        unit (str): Unit of measurement
        
    Returns:
        str: Formatted energy value
    """
    if value >= 1000:
        return f"{value/1000:.1f} M{unit}"
    elif value >= 1:
        return f"{value:.2f} {unit}"
    else:
        return f"{value*1000:.1f} m{unit}"

def get_energy_insights(df):
    """
    Generate basic energy usage insights.
    
    Args:
        df (pd.DataFrame): Energy data
        
    Returns:
        dict: Dictionary of insights
    """
    daily_stats = calculate_daily_stats(df)
    
    insights = {
        'total_energy': df['energy_kwh'].sum(),
        'avg_daily_energy': daily_stats['total_kwh'].mean(),
        'peak_day': daily_stats.loc[daily_stats['total_kwh'].idxmax(), 'date'],
        'peak_usage': daily_stats['total_kwh'].max(),
        'lowest_day': daily_stats.loc[daily_stats['total_kwh'].idxmin(), 'date'],
        'lowest_usage': daily_stats['total_kwh'].min(),
        'total_days': len(daily_stats),
        'avg_hourly_usage': df['energy_kwh'].mean(),
        'peak_hour': df.groupby('hour')['energy_kwh'].mean().idxmax(),
        'weekend_avg': df[df['day_of_week'] >= 5]['energy_kwh'].mean(),
        'weekday_avg': df[df['day_of_week'] < 5]['energy_kwh'].mean()
    }
    
    return insights

def create_sample_data_if_needed():
    """
    Create sample data if it doesn't exist.
    
    Returns:
        str: Path to sample data file
    """
    sample_file = 'data/sample_data.csv'
    
    if not os.path.exists(sample_file):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Import and run data generator
        from src.data_generator import create_sample_dataset
        create_sample_dataset()
    
    return sample_file

def load_data(file_path):
    """
    Load energy data from file.
    
    Args:
        file_path (str): Path to data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        df = process_uploaded_data(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def save_data(df, file_path):
    """
    Save energy data to file.
    
    Args:
        df (pd.DataFrame): Data to save
        file_path (str): Path to save file
    """
    try:
        df.to_csv(file_path, index=False)
        st.success(f"Data saved successfully to {file_path}")
    except Exception as e:
        st.error(f"Error saving data: {str(e)}") 