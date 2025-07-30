#!/usr/bin/env python3
"""
Test form submission to see what's happening with the prediction processing.
"""

import os
import sys
import requests
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from django.test import Client
from django.urls import reverse

def test_form_submission():
    """Test form submission to see what's happening."""
    print("=" * 60)
    print("TESTING FORM SUBMISSION")
    print("=" * 60)
    
    client = Client()
    
    # Test cases
    test_cases = [
        ("Man City", "Fulham", "European Leagues"),
        ("Arsenal", "Chelsea", "European Leagues"),
        ("Grasshoppers", "Lugano", "Others"),
    ]
    
    for home_team, away_team, category in test_cases:
        print(f"\n--- Testing form submission: {home_team} vs {away_team} ({category}) ---")
        
        try:
            # Simulate form submission
            response = client.post('/predict/', {
                'home_team': home_team,
                'away_team': away_team,
                'category': category,
            })
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                print("✓ Form submission successful")
                
                # Check if we got redirected to result page
                if hasattr(response, 'url') and response.url:
                    if 'result' in response.url:
                        print(f"✓ Redirected to result page: {response.url}")
                    else:
                        print(f"Response URL: {response.url}")
                else:
                    print("Response: No redirect (direct render)")
                    
                # Check response content for prediction info
                content = response.content.decode('utf-8')
                
                # Look for prediction indicators
                if 'Fallback prediction' in content:
                    print("⚠️  Found fallback prediction in response")
                elif 'Model 1 Details' in content:
                    print("✓ Found model details in response")
                else:
                    print("⚠️  No clear prediction indicators found")
                    
            else:
                print(f"✗ Form submission failed with status {response.status_code}")
                
        except Exception as e:
            print(f"✗ Form submission error: {e}")
            # Continue with next test case instead of crashing

def test_direct_prediction():
    """Test direct prediction without form."""
    print("\n" + "=" * 60)
    print("TESTING DIRECT PREDICTION")
    print("=" * 60)
    
    from predictor.views import predict
    from django.http import HttpRequest
    from django.contrib.auth.models import AnonymousUser
    
    # Create a mock request
    request = HttpRequest()
    request.method = 'POST'
    request.POST = {
        'home_team': 'Man City',
        'away_team': 'Fulham',
        'category': 'European Leagues',
    }
    request.user = AnonymousUser()
    
    try:
        response = predict(request)
        print(f"Response status: {response.status_code}")
        print(f"Response URL: {response.url if hasattr(response, 'url') else 'No redirect'}")
        
    except Exception as e:
        print(f"✗ Direct prediction error: {e}")

def main():
    """Run the tests."""
    print("FORM SUBMISSION TEST")
    print("=" * 60)
    
    # Test 1: Form submission
    test_form_submission()
    
    # Test 2: Direct prediction
    test_direct_prediction()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 