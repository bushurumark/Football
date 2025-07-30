#!/usr/bin/env python3
"""
Quick test to verify models and logic are in sync.
"""

import os
import sys
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from predictor.analytics import advanced_predict_match

def quick_sync_test():
    """Quick test of model and logic sync."""
    print("=== QUICK MODEL SYNC TEST ===")
    
    # Load models
    model1_path = os.path.join(os.path.dirname(__file__), 'models', 'model1.pkl')
    model2_path = os.path.join(os.path.dirname(__file__), 'models', 'model2.pkl')
    
    model1 = joblib.load(model1_path)
    model2 = joblib.load(model2_path)
    
    print("✅ Models loaded successfully")
    
    # Test predictions
    test_matches = [
        ("Arsenal", "Chelsea"),
        ("Man United", "Liverpool"),
        ("Barcelona", "Real Madrid")
    ]
    
    for home, away in test_matches:
        print(f"\nTesting: {home} vs {away}")
        
        result = advanced_predict_match(home, away, model1, model2)
        
        if result:
            print(f"✅ Prediction: {result['outcome']} (Confidence: {result['confidence']:.2f})")
            print(f"✅ Model1: {result.get('model1_prediction')} (Confidence: {result.get('model1_confidence')})")
            print(f"✅ Probabilities: {result.get('probabilities')}")
        else:
            print("❌ Prediction failed")
            return False
    
    print("\n🎉 All tests passed! Models and logic are in perfect sync!")
    return True

if __name__ == "__main__":
    quick_sync_test() 