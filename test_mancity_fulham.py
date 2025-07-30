#!/usr/bin/env python3
"""
Test specifically for Man City vs Fulham prediction.
"""

import os
import sys
import joblib
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from predictor.analytics import advanced_predict_match, load_football_data

def test_mancity_fulham():
    """Test Man City vs Fulham prediction specifically."""
    print("=" * 60)
    print("TESTING MAN CITY VS FULHAM PREDICTION")
    print("=" * 60)
    
    # Load models
    model1_path = os.path.join(project_root, 'models', 'model1.pkl')
    model2_path = os.path.join(project_root, 'models', 'model2.pkl')
    
    print(f"\nLoading models...")
    try:
        model1 = joblib.load(model1_path)
        print("✓ Model 1 loaded successfully")
        print(f"  - Type: {type(model1)}")
        print(f"  - Features: {model1.n_features_in_}")
        print(f"  - Classes: {model1.classes_}")
    except Exception as e:
        print(f"✗ Model 1 loading failed: {e}")
        return
    
    try:
        model2 = joblib.load(model2_path)
        print("✓ Model 2 loaded successfully")
        print(f"  - Type: {type(model2)}")
        print(f"  - Features: {model2.n_features_in_}")
    except Exception as e:
        print(f"✗ Model 2 loading failed: {e}")
        return
    
    # Test the specific prediction
    home_team = "Man City"
    away_team = "Fulham"
    
    print(f"\nTesting prediction: {home_team} vs {away_team}")
    
    try:
        result = advanced_predict_match(home_team, away_team, model1, model2)
        
        if result:
            print("✓ Prediction successful")
            print(f"  - Outcome: {result.get('outcome', 'Unknown')}")
            print(f"  - Prediction Number: {result.get('prediction_number', 'Unknown')}")
            print(f"  - Confidence: {result.get('model1_confidence', 0):.3f}")
            
            # Show probabilities
            probs = result.get('probabilities', {})
            if probs:
                print(f"  - Probabilities:")
                for outcome, prob in probs.items():
                    print(f"    {outcome}: {prob:.3f}")
            
            # Show basis
            basis = result.get('model1_basis', 'Unknown')
            print(f"  - Basis: {basis}")
            
            # Show model details
            model1_pred = result.get('model1_prediction', 'Unknown')
            model1_probs = result.get('model1_probs', {})
            print(f"  - Model 1 Prediction: {model1_pred}")
            if model1_probs:
                print(f"  - Model 1 Probabilities: {model1_probs}")
            
        else:
            print("✗ Prediction failed - result is None")
            
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def test_data_availability():
    """Test if data is available for Man City vs Fulham."""
    print("\n" + "=" * 60)
    print("TESTING DATA AVAILABILITY")
    print("=" * 60)
    
    data = load_football_data()
    if data is None:
        print("✗ No football data available")
        return
    
    print(f"✓ Football data loaded: {len(data)} rows")
    
    # Check for Man City matches
    mancity_matches = data[data['HomeTeam'] == 'Man City']
    print(f"✓ Man City home matches: {len(mancity_matches)}")
    
    # Check for Fulham matches
    fulham_matches = data[data['AwayTeam'] == 'Fulham']
    print(f"✓ Fulham away matches: {len(fulham_matches)}")
    
    # Check head-to-head
    h2h = data[(data['HomeTeam'] == 'Man City') & (data['AwayTeam'] == 'Fulham')]
    print(f"✓ Head-to-head matches: {len(h2h)}")
    
    if len(h2h) > 0:
        print("Sample head-to-head data:")
        print(h2h[['Date', 'FTHG', 'FTAG', 'FTR']].head())
    else:
        print("No head-to-head data found")
        
        # Check if teams exist in data
        all_teams = set(data['HomeTeam'].unique()).union(set(data['AwayTeam'].unique()))
        print(f"\nAll teams in dataset: {len(all_teams)}")
        
        if 'Man City' in all_teams:
            print("✓ Man City found in dataset")
        else:
            print("✗ Man City NOT found in dataset")
            
        if 'Fulham' in all_teams:
            print("✓ Fulham found in dataset")
        else:
            print("✗ Fulham NOT found in dataset")

def test_team_categories():
    """Test team categorization."""
    print("\n" + "=" * 60)
    print("TESTING TEAM CATEGORIZATION")
    print("=" * 60)
    
    # Check if teams are in the correct categories
    from predictor.views import LEAGUES_BY_CATEGORY
    
    main_teams = set()
    for league_teams in LEAGUES_BY_CATEGORY['European Leagues'].values():
        main_teams.update(league_teams)
    
    other_teams = set()
    for league_teams in LEAGUES_BY_CATEGORY['Others'].values():
        other_teams.update(league_teams)
    
    print(f"Main teams count: {len(main_teams)}")
    print(f"Other teams count: {len(other_teams)}")
    
    if 'Man City' in main_teams:
        print("✓ Man City in 'European Leagues' category")
    else:
        print("✗ Man City NOT in 'European Leagues' category")
        
    if 'Fulham' in main_teams:
        print("✓ Fulham in 'European Leagues' category")
    else:
        print("✗ Fulham NOT in 'European Leagues' category")

def main():
    """Run the tests."""
    print("MAN CITY VS FULHAM PREDICTION TEST")
    print("=" * 60)
    
    # Test 1: Team categorization
    test_team_categories()
    
    # Test 2: Data availability
    test_data_availability()
    
    # Test 3: Specific prediction
    test_mancity_fulham()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 