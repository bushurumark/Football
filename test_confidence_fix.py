#!/usr/bin/env python3
"""
Test to verify confidence calculation fix.
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

def test_confidence_fix():
    """Test that confidence scores are now realistic."""
    print("=== TESTING CONFIDENCE FIX ===")
    
    # Load models
    model1_path = os.path.join(os.path.dirname(__file__), 'models', 'model1.pkl')
    model1 = joblib.load(model1_path)
    
    # Test predictions
    test_matches = [
        ("Arsenal", "Chelsea"),
        ("Man United", "Liverpool"),
        ("Barcelona", "Real Madrid"),
        ("Bayern Munich", "Dortmund"),
        ("Bournemouth", "Brighton")
    ]
    
    print("\nTesting confidence scores:")
    print("-" * 50)
    
    for home, away in test_matches:
        print(f"\n{home} vs {away}:")
        
        result = advanced_predict_match(home, away, model1)
        
        if result:
            confidence = result['confidence']
            prediction = result['outcome']
            probs = result['probabilities']
            
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.2f} ({confidence*100:.1f}%)")
            print(f"  Probabilities: {probs}")
            
            # Check if confidence is realistic
            if confidence > 0.95:
                print(f"  ⚠️  WARNING: Confidence too high ({confidence:.2f})")
            elif confidence < 0.1:
                print(f"  ⚠️  WARNING: Confidence too low ({confidence:.2f})")
            else:
                print(f"  ✅ Confidence is realistic ({confidence:.2f})")
        else:
            print(f"  ❌ Prediction failed")
    
    print("\n" + "=" * 50)
    print("Confidence fix test completed!")

if __name__ == "__main__":
    test_confidence_fix() 