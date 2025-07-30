#!/usr/bin/env python3
"""
Test to verify Model 2 is working properly.
"""

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from predictor.analytics import advanced_predict_match, load_football_data
import joblib

def test_model2():
    """Test that Model 2 is working properly."""
    print("=== TESTING MODEL 2 ===")
    
    # Load models
    model1_path = os.path.join(os.path.dirname(__file__), 'models', 'model1.pkl')
    model2_path = os.path.join(os.path.dirname(__file__), 'models', 'model2.pkl')
    
    model1 = joblib.load(model1_path)
    model2 = joblib.load(model2_path)
    
    print("✅ Models loaded successfully")
    print(f"✅ Model 1 type: {type(model1)}")
    print(f"✅ Model 2 type: {type(model2)}")
    
    # Test predictions
    test_matches = [
        ("Arsenal", "Chelsea"),
        ("Man United", "Liverpool"),
        ("Barcelona", "Real Madrid"),
        ("Bayern Munich", "Dortmund"),
        ("Bournemouth", "Brighton")
    ]
    
    print("\nTesting Model 2 predictions:")
    print("-" * 50)
    
    for home, away in test_matches:
        print(f"\n{home} vs {away}:")
        
        result = advanced_predict_match(home, away, model1, model2)
        
        if result:
            # Model 1 details
            model1_pred = result.get('model1_prediction')
            model1_conf = result.get('model1_confidence')
            model1_probs = result.get('model1_probs')
            
            # Model 2 details
            model2_pred = result.get('model2_prediction')
            model2_conf = result.get('model2_confidence')
            model2_probs = result.get('model2_probs')
            
            print(f"  Model 1: {model1_pred} (Confidence: {model1_conf:.2f})")
            print(f"  Model 2: {model2_pred} (Confidence: {model2_conf:.2f})")
            
            if model1_probs:
                print(f"  Model 1 Probabilities: {model1_probs}")
            if model2_probs:
                print(f"  Model 2 Probabilities: {model2_probs}")
            
            # Check if Model 2 is working
            if model2_pred is not None and model2_conf is not None:
                print(f"  ✅ Model 2 is working!")
            else:
                print(f"  ❌ Model 2 not working properly")
        else:
            print(f"  ❌ Prediction failed")
    
    print("\n" + "=" * 50)
    print("Model 2 test completed!")

if __name__ == "__main__":
    test_model2() 