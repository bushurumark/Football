#!/usr/bin/env python3
"""
Test the web interface to verify it's working correctly.
"""

import requests
import json
import time

def test_web_api():
    """Test the web API endpoint."""
    print("=" * 60)
    print("TESTING WEB INTERFACE")
    print("=" * 60)
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Test cases
    test_cases = [
        ("Fulham", "Man United", "European Leagues"),
        ("Grasshoppers", "Lugano", "Others"),
        ("Arsenal", "Chelsea", "European Leagues"),
    ]
    
    for home_team, away_team, category in test_cases:
        print(f"\n--- Testing: {home_team} vs {away_team} ({category}) ---")
        
        try:
            # Test API endpoint
            url = "http://127.0.0.1:8000/api/predict/"
            data = {
                "home_team": home_team,
                "away_team": away_team,
                "category": category
            }
            
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print("✓ API endpoint working")
                print(f"  - Prediction: {result.get('outcome', 'Unknown')}")
                print(f"  - Scores: {result.get('home_score', '?')} - {result.get('away_score', '?')}")
                print(f"  - Prediction Number: {result.get('prediction_number', 'Unknown')}")
                
                # Show probabilities
                probs = result.get('probabilities', {})
                if probs:
                    print(f"  - Probabilities:")
                    for outcome, prob in probs.items():
                        print(f"    {outcome}: {prob:.1f}%")
            else:
                print(f"✗ API endpoint returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("⚠️  Server not running. Start with: python manage.py runserver")
            break
        except Exception as e:
            print(f"✗ API test failed: {e}")

def test_web_page():
    """Test the main web page."""
    print("\n" + "=" * 60)
    print("TESTING WEB PAGE")
    print("=" * 60)
    
    try:
        # Test main page
        url = "http://127.0.0.1:8000/"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print("✓ Main page accessible")
        else:
            print(f"✗ Main page returned status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running")
    except Exception as e:
        print(f"✗ Web page test failed: {e}")

def main():
    """Run the tests."""
    print("WEB INTERFACE TEST")
    print("=" * 60)
    
    # Test 1: Web page
    test_web_page()
    
    # Test 2: API endpoint
    test_web_api()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nTo test manually:")
    print("1. Open: http://127.0.0.1:8000/")
    print("2. Try making predictions!")
    print("3. Check if the model predictions are showing correctly")

if __name__ == "__main__":
    main() 