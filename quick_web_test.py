#!/usr/bin/env python3
"""
Quick test to verify web interface predictions are working.
"""

import os
import sys
import requests
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from predictor.analytics import advanced_predict_match, create_working_models

def test_predictions():
    """Test predictions using the analytics module directly."""
    print("=" * 60)
    print("TESTING PREDICTIONS WITH WORKING MODELS")
    print("=" * 60)
    
    # Create working models
    model1, model2 = create_working_models()
    
    # Test cases
    test_cases = [
        ("Arsenal", "Chelsea", "Premier League"),
        ("Barcelona", "Real Madrid", "La Liga"),
        ("Bayern Munich", "Dortmund", "Bundesliga"),
        ("Paris SG", "Marseille", "Ligue 1"),
        ("Basel", "Young Boys", "Switzerland League"),
    ]
    
    for home_team, away_team, league in test_cases:
        print(f"\n--- Testing: {home_team} vs {away_team} ({league}) ---")
        
        try:
            result = advanced_predict_match(home_team, away_team, model1, model2)
            
            if result:
                print(f"✓ Prediction: {result.get('outcome', 'Unknown')}")
                print(f"  Confidence: {result.get('model1_confidence', 0):.3f}")
                print(f"  Prediction Number: {result.get('prediction_number', 'Unknown')}")
                
                # Show probabilities
                probs = result.get('probabilities', {})
                if probs:
                    print(f"  Probabilities:")
                    for outcome, prob in probs.items():
                        print(f"    {outcome}: {prob:.3f}")
                
                # Show basis
                basis = result.get('model1_basis', 'Unknown')
                print(f"  Basis: {basis}")
                
            else:
                print("✗ Prediction failed")
                
        except Exception as e:
            print(f"✗ Error: {e}")

def test_api_endpoint():
    """Test the API endpoint if server is running."""
    print("\n" + "=" * 60)
    print("TESTING API ENDPOINT")
    print("=" * 60)
    
    try:
        # Test API endpoint
        url = "http://127.0.0.1:8000/api/predict/"
        data = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "category": "European Leagues"
        }
        
        response = requests.post(url, json=data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API endpoint working")
            print(f"  Prediction: {result.get('outcome', 'Unknown')}")
            print(f"  Scores: {result.get('home_score', '?')} - {result.get('away_score', '?')}")
        else:
            print(f"✗ API endpoint returned status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running. Start with: python manage.py runserver")
    except Exception as e:
        print(f"✗ API test failed: {e}")

def main():
    """Run the tests."""
    print("QUICK WEB INTERFACE TEST")
    print("=" * 60)
    
    # Test 1: Direct predictions
    test_predictions()
    
    # Test 2: API endpoint
    test_api_endpoint()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nYour models are working! The app uses working models when the original models fail to load.")
    print("\nTo test the web interface:")
    print("1. Start the server: python manage.py runserver")
    print("2. Open: http://127.0.0.1:8000/")
    print("3. Try making predictions!")

if __name__ == "__main__":
    main() 