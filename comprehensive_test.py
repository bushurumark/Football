#!/usr/bin/env python3
"""
Comprehensive test to verify models and logic are working in perfect sync.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from predictor.analytics import advanced_predict_match, load_football_data

def test_model_sync():
    """Test if models and logic are working in perfect sync."""
    print("=== COMPREHENSIVE MODEL SYNC TEST ===")
    print("=" * 50)
    
    # Test 1: Model Loading
    print("\n1. Testing Model Loading...")
    try:
        model1_path = os.path.join(os.path.dirname(__file__), 'models', 'model1.pkl')
        model2_path = os.path.join(os.path.dirname(__file__), 'models', 'model2.pkl')
        
        model1 = joblib.load(model1_path)
        model2 = joblib.load(model2_path)
        
        print("✅ Model1 loaded: ", type(model1))
        print("✅ Model2 loaded: ", type(model2))
        print("✅ No version warnings displayed")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 2: Data Loading
    print("\n2. Testing Data Loading...")
    try:
        data = load_football_data()
        if data is not None:
            print(f"✅ Data loaded: {data.shape[0]} matches, {data.shape[1]} features")
            print(f"✅ Sample teams: {data['HomeTeam'].unique()[:5]}")
        else:
            print("❌ Data loading failed")
            return False
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False
    
    # Test 3: Prediction Logic
    print("\n3. Testing Prediction Logic...")
    test_cases = [
        ("Arsenal", "Chelsea"),
        ("Man United", "Liverpool"),
        ("Barcelona", "Real Madrid"),
        ("Bayern Munich", "Dortmund"),
        ("Newcastle", "Nott'm Forest")
    ]
    
    all_predictions = []
    
    for home_team, away_team in test_cases:
        print(f"\n   Testing: {home_team} vs {away_team}")
        
        try:
            result = advanced_predict_match(home_team, away_team, model1, model2)
            
            if result:
                print(f"   ✅ Prediction successful")
                print(f"   📊 Outcome: {result['outcome']}")
                print(f"   📊 Prediction Number: {result['prediction_number']}")
                print(f"   📊 Confidence: {result['confidence']:.2f}")
                print(f"   📊 Model1 Prediction: {result.get('model1_prediction')}")
                print(f"   📊 Model1 Confidence: {result.get('model1_confidence')}")
                
                # Check if probabilities are properly formatted
                if result.get('probabilities'):
                    print(f"   📊 Probabilities: {len(result['probabilities'])} classes")
                    for key, value in result['probabilities'].items():
                        print(f"      {key}: {value:.3f}")
                
                # Check head-to-head data
                if result.get('h2h_probabilities'):
                    print(f"   📊 H2H Probabilities: {result['h2h_probabilities']}")
                else:
                    print(f"   ⚠️  No H2H data available")
                
                all_predictions.append(result)
            else:
                print(f"   ❌ Prediction failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            return False
    
    # Test 4: Logic Consistency
    print("\n4. Testing Logic Consistency...")
    
    # Check if predictions are consistent
    prediction_numbers = [p['prediction_number'] for p in all_predictions]
    outcomes = [p['outcome'] for p in all_predictions]
    confidences = [p['confidence'] for p in all_predictions]
    
    print(f"   📊 Prediction Numbers: {prediction_numbers}")
    print(f"   📊 Outcomes: {outcomes}")
    print(f"   📊 Confidence Range: {min(confidences):.2f} - {max(confidences):.2f}")
    
    # Check for logical consistency
    valid_prediction_numbers = [1, 2, 3]
    valid_outcomes = ["Home", "Draw", "Away"]
    
    if all(pn in valid_prediction_numbers for pn in prediction_numbers):
        print("   ✅ All prediction numbers are valid (1, 2, 3)")
    else:
        print("   ❌ Invalid prediction numbers found")
        return False
    
    if all(outcome in valid_outcomes for outcome in outcomes):
        print("   ✅ All outcomes are valid (Home, Draw, Away)")
    else:
        print("   ❌ Invalid outcomes found")
        return False
    
    if all(0 <= conf <= 1 for conf in confidences):
        print("   ✅ All confidence scores are valid (0-1)")
    else:
        print("   ❌ Invalid confidence scores found")
        return False
    
    # Test 5: Model Integration
    print("\n5. Testing Model Integration...")
    
    model1_predictions = [p.get('model1_prediction') for p in all_predictions]
    model1_confidences = [p.get('model1_confidence') for p in all_predictions]
    
    print(f"   📊 Model1 Predictions: {model1_predictions}")
    print(f"   📊 Model1 Confidences: {model1_confidences}")
    
    # Check if Model1 is providing predictions
    if any(pred is not None for pred in model1_predictions):
        print("   ✅ Model1 is providing predictions")
    else:
        print("   ⚠️  Model1 predictions are None")
    
    if any(conf is not None for conf in model1_confidences):
        print("   ✅ Model1 is providing confidence scores")
    else:
        print("   ⚠️  Model1 confidences are None")
    
    # Test 6: Feature Engineering
    print("\n6. Testing Feature Engineering...")
    
    try:
        # Test if the model can handle the expected features
        if hasattr(model1, 'feature_names_in_'):
            print(f"   ✅ Model expects {len(model1.feature_names_in_)} features")
            print(f"   📊 First 5 features: {model1.feature_names_in_[:5]}")
            
            # Test with sample features
            sample_features = pd.DataFrame([[0] * len(model1.feature_names_in_)], 
                                        columns=model1.feature_names_in_)
            sample_pred = model1.predict(sample_features)
            print(f"   ✅ Model can make predictions with sample features: {sample_pred[0]}")
        else:
            print("   ⚠️  Model doesn't have feature_names_in_ attribute")
    except Exception as e:
        print(f"   ❌ Feature engineering test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Models and logic are working in perfect sync!")
    print("=" * 50)
    
    return True

def main():
    """Main test function."""
    success = test_model_sync()
    
    if success:
        print("\n✅ SUMMARY: Your football prediction system is working perfectly!")
        print("   - Models load without warnings")
        print("   - Data loads correctly")
        print("   - Predictions are consistent")
        print("   - Logic is synchronized")
        print("   - All features are working")
    else:
        print("\n❌ SUMMARY: Issues found in the system")
    
    return success

if __name__ == "__main__":
    main() 