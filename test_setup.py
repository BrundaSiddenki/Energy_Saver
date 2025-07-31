"""
Test script for SmartEnergySense Week 1 setup
Verifies that all components are working correctly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✅ plotly imported successfully")
    except ImportError as e:
        print(f"❌ plotly import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    try:
        from src.data_generator import EnergyDataGenerator
        print("✅ EnergyDataGenerator imported successfully")
    except ImportError as e:
        print(f"❌ EnergyDataGenerator import failed: {e}")
        return False
    
    try:
        from utils.helpers import validate_energy_data, process_uploaded_data
        print("✅ Helper functions imported successfully")
    except ImportError as e:
        print(f"❌ Helper functions import failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test data generation functionality."""
    print("\n🧪 Testing data generation...")
    
    try:
        from src.data_generator import EnergyDataGenerator
        
        # Create a small test dataset
        generator = EnergyDataGenerator(days=7)
        df = generator.generate_sample_data()
        
        print(f"✅ Generated {len(df)} hours of data")
        print(f"✅ Data shape: {df.shape}")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"✅ Average daily usage: {df.groupby('date')['energy_kwh'].sum().mean():.2f} kWh")
        
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_helper_functions():
    """Test helper functions."""
    print("\n🧪 Testing helper functions...")
    
    try:
        from utils.helpers import validate_energy_data, process_uploaded_data, get_energy_insights
        from src.data_generator import EnergyDataGenerator
        
        # Generate test data
        generator = EnergyDataGenerator(days=7)
        df = generator.generate_sample_data()
        
        # Test validation
        is_valid, msg = validate_energy_data(df)
        print(f"✅ Data validation: {msg}")
        
        # Test processing
        processed_df = process_uploaded_data(df)
        print(f"✅ Data processing successful, shape: {processed_df.shape}")
        
        # Test insights
        insights = get_energy_insights(processed_df)
        print(f"✅ Generated insights: {len(insights)} metrics")
        
        return True
    except Exception as e:
        print(f"❌ Helper functions test failed: {e}")
        return False

def test_visualization():
    """Test visualization capabilities."""
    print("\n🧪 Testing visualization...")
    
    try:
        import plotly.express as px
        from src.data_generator import EnergyDataGenerator
        
        # Generate test data
        generator = EnergyDataGenerator(days=7)
        df = generator.generate_sample_data()
        
        # Create a simple chart
        fig = px.line(df, x='timestamp', y='energy_kwh', title='Test Chart')
        print("✅ Plotly chart created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 SmartEnergySense Week 1 Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_generation,
        test_helper_functions,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Week 1 setup is complete.")
        print("\n🚀 You can now run the application with:")
        print("   streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 