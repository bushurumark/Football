#!/usr/bin/env python3
"""
Comprehensive test script for the football prediction models.
Tests both model1.pkl and model2.pkl with various scenarios.
"""

import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from predictor.analytics import advanced_predict_match, create_working_models, load_football_data
from predictor.models import Team, Match

def test_model_loading():
    """Test if models can be loaded successfully."""
    print("=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    
    model1_path = os.path.join(project_root, 'models', 'model1.pkl')
    model2_path = os.path.join(project_root, 'models', 'model2.pkl')
    
    # Test Model 1
    print(f"\nTesting Model 1: {model1_path}")
    try:
        with open(model1_path, 'rb') as f:
            model1 = pickle.load(f)
        print("✓ Model 1 loaded successfully with pickle")
        
        # Test with joblib as well
        model1_joblib = joblib.load(model1_path)
        print("✓ Model 1 loaded successfully with joblib")
        
        # Test model properties
        print(f"  - Model type: {type(model1)}")
        if hasattr(model1, 'feature_names_in_'):
            print(f"  - Expected features: {len(model1.feature_names_in_)}")
        if hasattr(model1, 'classes_'):
            print(f"  - Classes: {model1.classes_}")
        
    except Exception as e:
        print(f"✗ Model 1 loading failed: {e}")
        model1 = None
    
    # Test Model 2
    print(f"\nTesting Model 2: {model2_path}")
    try:
        with open(model2_path, 'rb') as f:
            model2 = pickle.load(f)
        print("✓ Model 2 loaded successfully with pickle")
        
        # Test with joblib as well
        model2_joblib = joblib.load(model2_path)
        print("✓ Model 2 loaded successfully with joblib")
        
        # Test model properties
        print(f"  - Model type: {type(model2)}")
        if hasattr(model2, 'feature_names_in_'):
            print(f"  - Expected features: {len(model2.feature_names_in_)}")
        
    except Exception as e:
        print(f"✗ Model 2 loading failed: {e}")
        model2 = None
    
    return model1, model2

def test_data_loading():
    """Test if football data can be loaded."""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    try:
        data = load_football_data()
        if data is not None:
            print("✓ Football data loaded successfully")
            print(f"  - Total rows: {len(data)}")
            print(f"  - Columns: {list(data.columns)}")
            
            # Show sample data
            print("\nSample data:")
            print(data.head(3))
            
            return data
        else:
            print("✗ No football data found")
            return None
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None

def test_prediction_scenarios(model1, model2, data):
    """Test predictions with various scenarios."""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION SCENARIOS")
    print("=" * 60)
    
    # Test scenarios
    test_cases = [
        # European League teams (should use Model 1)
        ("Arsenal", "Chelsea", "European Leagues"),
        ("Barcelona", "Real Madrid", "European Leagues"),
        ("Bayern Munich", "Dortmund", "European Leagues"),
        ("Paris SG", "Marseille", "European Leagues"),
        
        # Other League teams (should use Model 2)
        ("Basel", "Young Boys", "Others"),
        ("Aarhus", "FC Copenhagen", "Others"),
        ("Grazer AK", "Salzburg", "Others"),
        
        # Mixed teams (should show error)
        ("Arsenal", "Basel", "Mixed"),
    ]
    
    for home_team, away_team, category in test_cases:
        print(f"\n--- Testing: {home_team} vs {away_team} ({category}) ---")
        
        try:
            # Test with both models
            result = advanced_predict_match(home_team, away_team, model1, model2)
            
            if result:
                print(f"✓ Prediction successful")
                print(f"  - Outcome: {result.get('outcome', 'Unknown')}")
                print(f"  - Prediction Number: {result.get('prediction_number', 'Unknown')}")
                print(f"  - Confidence: {result.get('model1_confidence', 'Unknown'):.3f}")
                print(f"  - Model 1 Prediction: {result.get('model1_prediction', 'Unknown')}")
                print(f"  - Model 2 Prediction: {result.get('model2_prediction', 'Unknown')}")
                
                # Show probabilities
                probs = result.get('probabilities', {})
                if probs:
                    print(f"  - Probabilities:")
                    for outcome, prob in probs.items():
                        print(f"    * {outcome}: {prob:.3f}")
                
                # Show basis
                basis = result.get('model1_basis', 'Unknown')
                print(f"  - Basis: {basis}")
                
            else:
                print(f"✗ Prediction failed")
                
        except Exception as e:
            print(f"✗ Error during prediction: {e}")

def test_working_models():
    """Test the fallback working models."""
    print("\n" + "=" * 60)
    print("TESTING WORKING MODELS (FALLBACK)")
    print("=" * 60)
    
    try:
        model1, model2 = create_working_models()
        print("✓ Working models created successfully")
        
        # Test predictions with working models
        test_teams = [("Arsenal", "Chelsea"), ("Barcelona", "Real Madrid")]
        
        for home_team, away_team in test_teams:
            print(f"\n--- Testing working models: {home_team} vs {away_team} ---")
            
            result = advanced_predict_match(home_team, away_team, model1, model2)
            
            if result:
                print(f"✓ Working model prediction successful")
                print(f"  - Outcome: {result.get('outcome', 'Unknown')}")
                print(f"  - Confidence: {result.get('model1_confidence', 'Unknown'):.3f}")
            else:
                print(f"✗ Working model prediction failed")
                
    except Exception as e:
        print(f"✗ Working models test failed: {e}")

def test_database_models():
    """Test database models and data."""
    print("\n" + "=" * 60)
    print("TESTING DATABASE MODELS")
    print("=" * 60)
    
    try:
        # Test Team model
        teams_count = Team.objects.count()
        print(f"✓ Teams in database: {teams_count}")
        
        if teams_count > 0:
            sample_teams = Team.objects.all()[:5]
            print("Sample teams:")
            for team in sample_teams:
                print(f"  - {team.name} ({team.league})")
        
        # Test Match model
        matches_count = Match.objects.count()
        print(f"✓ Matches in database: {matches_count}")
        
        if matches_count > 0:
            sample_matches = Match.objects.all()[:3]
            print("Sample matches:")
            for match in sample_matches:
                print(f"  - {match.home_team} vs {match.away_team} ({match.date})")
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")

def test_api_endpoints():
    """Test API endpoints."""
    print("\n" + "=" * 60)
    print("TESTING API ENDPOINTS")
    print("=" * 60)
    
    # This would require a running server, so we'll just check the URL patterns
    try:
        from predictor.urls import urlpatterns
        print("✓ URL patterns found:")
        for pattern in urlpatterns:
            print(f"  - {pattern.pattern}")
    except Exception as e:
        print(f"✗ URL pattern test failed: {e}")

def main():
    """Run all tests."""
    print("FOOTBALL PREDICTION APP - COMPREHENSIVE MODEL TESTING")
    print("=" * 60)
    
    # Test 1: Model Loading
    model1, model2 = test_model_loading()
    
    # Test 2: Data Loading
    data = test_data_loading()
    
    # Test 3: Prediction Scenarios
    if model1 or model2:
        test_prediction_scenarios(model1, model2, data)
    
    # Test 4: Working Models
    test_working_models()
    
    # Test 5: Database Models
    test_database_models()
    
    # Test 6: API Endpoints
    test_api_endpoints()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\nTo test the web interface:")
    print("1. Make sure the server is running: python manage.py runserver")
    print("2. Open http://127.0.0.1:8000/ in your browser")
    print("3. Try making predictions with different teams")

if __name__ == "__main__":
    main() 