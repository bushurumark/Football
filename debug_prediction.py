#!/usr/bin/env python3
"""
Debug script to test prediction and see what data is returned.
"""

import os
import sys
import joblib

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from predictor.analytics import advanced_predict_match, load_football_data

def debug_prediction():
    """Debug a specific prediction."""
    print("=== Debugging Prediction ===")
    
    # Load models
    model1_path = os.path.join(os.path.dirname(__file__), 'models', 'model1.pkl')
    model2_path = os.path.join(os.path.dirname(__file__), 'models', 'model2.pkl')
    
    model1 = joblib.load(model1_path)
    model2 = joblib.load(model2_path)
    
    # Test prediction
    home_team = "Man City"
    away_team = "Newcastle"
    
    print(f"Testing: {home_team} vs {away_team}")
    
    result = advanced_predict_match(home_team, away_team, model1, model2)
    
    if result:
        print("\n=== RESULT DATA ===")
        for key, value in result.items():
            print(f"{key}: {value}")
        
        print("\n=== MODEL DETAILS ===")
        print(f"model1_confidence: {result.get('model1_confidence')}")
        print(f"model1_probs: {result.get('model1_probs')}")
        print(f"model1_prediction: {result.get('model1_prediction')}")
        
        print("\n=== PROBABILITIES FORMAT ===")
        if result.get('model1_probs'):
            print("Model1 probabilities:")
            for k, v in result['model1_probs'].items():
                print(f"  {k}: {v}")
        else:
            print("No model1_probs found")
            
    else:
        print("❌ Prediction failed")

if __name__ == "__main__":
    debug_prediction() 