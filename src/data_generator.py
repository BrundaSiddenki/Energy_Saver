"""
Data Generator for SmartEnergySense
Generates realistic sample energy usage data for demonstration and testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class EnergyDataGenerator:
    """Generates realistic household energy usage data with patterns."""
    
    def __init__(self, start_date=None, days=90):
        """
        Initialize the data generator.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            days (int): Number of days to generate data for
        """
        self.start_date = start_date or (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        self.days = days
        
    def generate_base_pattern(self):
        """Generate base daily energy usage pattern."""
        # 24-hour pattern with peaks during morning and evening
        hours = np.arange(24)
        
        # Morning peak (6-9 AM)
        morning_peak = 0.8 * np.exp(-0.5 * ((hours - 7) / 2) ** 2)
        
        # Evening peak (6-10 PM)
        evening_peak = 1.0 * np.exp(-0.5 * ((hours - 20) / 2) ** 2)
        
        # Night baseline (midnight to 6 AM)
        night_baseline = 0.3 * np.ones_like(hours)
        night_baseline[hours < 6] = 0.2
        
        # Day baseline (10 AM to 6 PM)
        day_baseline = 0.5 * np.ones_like(hours)
        day_baseline[(hours >= 10) & (hours < 18)] = 0.6
        
        # Combine patterns
        base_pattern = morning_peak + evening_peak + night_baseline + day_baseline
        
        return base_pattern
    
    def add_weekly_pattern(self, base_data):
        """Add weekly patterns (weekend vs weekday differences)."""
        # Weekend days typically have higher usage
        weekend_multiplier = 1.2
        weekday_multiplier = 1.0
        
        # Add some randomness to make it realistic
        noise = np.random.normal(0, 0.1, len(base_data))
        
        for i in range(len(base_data)):
            day_of_week = (pd.to_datetime(self.start_date) + timedelta(days=i//24)).weekday()
            
            if day_of_week >= 5:  # Weekend (Saturday=5, Sunday=6)
                base_data[i] *= weekend_multiplier
            else:
                base_data[i] *= weekday_multiplier
                
            base_data[i] += noise[i]
            base_data[i] = max(0.1, base_data[i])  # Ensure positive values
            
        return base_data
    
    def add_seasonal_pattern(self, base_data):
        """Add seasonal patterns (higher usage in summer/winter)."""
        # Simulate seasonal variations
        for i in range(len(base_data)):
            day_of_year = (pd.to_datetime(self.start_date) + timedelta(days=i//24)).timetuple().tm_yday
            
            # Summer peak (June-August) and winter peak (December-February)
            summer_factor = 1.3 * np.exp(-0.5 * ((day_of_year - 180) / 30) ** 2)  # June 29
            winter_factor = 1.4 * np.exp(-0.5 * ((day_of_year - 15) / 30) ** 2)   # January 15
            
            seasonal_multiplier = 1.0 + summer_factor + winter_factor
            base_data[i] *= seasonal_multiplier
            
        return base_data
    
    def add_anomalies(self, base_data):
        """Add realistic anomalies (spikes, drops)."""
        # Add random spikes (5% chance per hour)
        for i in range(len(base_data)):
            if random.random() < 0.05:
                spike_factor = random.uniform(1.5, 3.0)
                base_data[i] *= spike_factor
                
        # Add occasional drops (2% chance per hour)
        for i in range(len(base_data)):
            if random.random() < 0.02:
                drop_factor = random.uniform(0.3, 0.7)
                base_data[i] *= drop_factor
                
        return base_data
    
    def generate_sample_data(self):
        """Generate complete sample energy usage dataset."""
        # Generate timestamps
        start_dt = pd.to_datetime(self.start_date)
        timestamps = pd.date_range(start=start_dt, periods=self.days*24, freq='H')
        
        # Generate base pattern
        base_pattern = self.generate_base_pattern()
        base_data = np.tile(base_pattern, self.days)
        
        # Add patterns
        base_data = self.add_weekly_pattern(base_data)
        base_data = self.add_seasonal_pattern(base_data)
        base_data = self.add_anomalies(base_data)
        
        # Convert to kWh (typical household usage: 10-30 kWh/day)
        # Scale to realistic values
        daily_avg = random.uniform(15, 25)  # kWh per day
        scaling_factor = daily_avg / np.mean(base_data)
        energy_usage = base_data * scaling_factor
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'energy_kwh': energy_usage,
            'date': timestamps.date,
            'hour': timestamps.hour,
            'day_of_week': timestamps.dayofweek,
            'month': timestamps.month
        })
        
        return df
    
    def save_sample_data(self, filename='data/sample_data.csv'):
        """Generate and save sample data to CSV."""
        df = self.generate_sample_data()
        df.to_csv(filename, index=False)
        print(f"Sample data saved to {filename}")
        print(f"Generated {len(df)} hours of data ({self.days} days)")
        print(f"Average daily usage: {df['energy_kwh'].sum() / self.days:.2f} kWh")
        return df

def create_sample_dataset():
    """Convenience function to create sample dataset."""
    generator = EnergyDataGenerator(days=90)  # 3 months of data
    return generator.save_sample_data()

if __name__ == "__main__":
    # Generate sample data when run directly
    create_sample_dataset() 