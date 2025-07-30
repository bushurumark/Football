#!/usr/bin/env python3
"""
Test script to check if the models are working correctly.
Run this script to test model loading and prediction functionality.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from predictor.analytics import advanced_predict_match, load_football_data

def test_model_loading():
    """Test if models can be loaded successfully."""
    print("=== Testing Model Loading ===")
    
    try:
        # Load models
        model1_path = os.path.join(os.path.dirname(__file__), 'models', 'model1.pkl')
        model2_path = os.path.join(os.path.dirname(__file__), 'models', 'model2.pkl')
        
        print(f"Loading model1 from: {model1_path}")
        model1 = joblib.load(model1_path)
        print(f"✓ Model1 loaded successfully. Type: {type(model1)}")
        
        print(f"Loading model2 from: {model2_path}")
        model2 = joblib.load(model2_path)
        print(f"✓ Model2 loaded successfully. Type: {type(model2)}")
        
        return model1, model2
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return None, None

def test_data_loading():
    """Test if data can be loaded successfully."""
    print("\n=== Testing Data Loading ===")
    
    try:
        data = load_football_data()
        if data is not None:
            print(f"✓ Data loaded successfully. Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print(f"Sample data:")
            print(data.head(3))
            return data
        else:
            print("✗ Failed to load data")
            return None
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def test_prediction(model1, data):
    """Test prediction functionality."""
    print("\n=== Testing Predictions ===")
    
    # Test teams
    test_cases = [
        ("Arsenal", "Chelsea"),
        ("Man United", "Liverpool"),
        ("Barcelona", "Real Madrid"),
        ("Bayern Munich", "Dortmund")
    ]
    
    for home_team, away_team in test_cases:
        print(f"\nTesting: {home_team} vs {away_team}")
        
        try:
            result = advanced_predict_match(home_team, away_team, model1)
            
            if result:
                print(f"✓ Prediction successful!")
                print(f"  Outcome: {result['outcome']}")
                print(f"  Confidence: {result['confidence']:.2f}")
                print(f"  Prediction Number: {result['prediction_number']}")
                print(f"  Final Prediction: {result['final_prediction']}")
                
                if result['h2h_probabilities']:
                    print(f"  Head-to-Head Probabilities: {result['h2h_probabilities']}")
                else:
                    print(f"  No head-to-head data available")
            else:
                print(f"✗ Prediction failed")
                
        except Exception as e:
            print(f"✗ Error making prediction: {e}")

def test_model_features(model1):
    """Test model feature requirements."""
    print("\n=== Testing Model Features ===")
    
    try:
        # Check what features the model expects
        if hasattr(model1, 'feature_names_in_'):
            print(f"✓ Model expects {len(model1.feature_names_in_)} features:")
            for i, feature in enumerate(model1.feature_names_in_):
                print(f"  {i+1}. {feature}")
        else:
            print("⚠ Model doesn't have feature_names_in_ attribute")
            
        # Test with sample features
        if hasattr(model1, 'feature_names_in_'):
            sample_features = pd.DataFrame([[0] * len(model1.feature_names_in_)], 
                                        columns=model1.feature_names_in_)
            prediction = model1.predict(sample_features)
            print(f"✓ Model can make predictions with sample features: {prediction}")
            
    except Exception as e:
        print(f"✗ Error testing model features: {e}")

def main():
    """Main test function."""
    print("Football Prediction Model Test")
    print("=" * 40)
    
    # Test model loading
    model1, model2 = test_model_loading()
    
    if model1 is None:
        print("\n❌ Cannot proceed without models. Exiting.")
        return
    
    # Test data loading
    data = test_data_loading()
    
    # Test model features
    test_model_features(model1)
    
    # Test predictions
    test_prediction(model1, data)
    
    print("\n" + "=" * 40)
    print("Test completed!")

if __name__ == "__main__":
    main() 