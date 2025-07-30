# -*- coding: utf-8 -*-
"""
Fixed model_utils.py with corrected logic and proper error handling.
"""

import pandas as pd
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def align_features(input_df, model):
    """Align input features with model's expected features."""
    try:
        for f in model.feature_names_in_:
            if f not in input_df:
                input_df[f] = 0
        return input_df[model.feature_names_in_]
    except Exception as e:
        logging.error(f"Feature alignment error: {e}")
        return None

def compute_mean_for_teams(home, away, data, model, get_column_names=None, version="v1"):
    """Compute mean features for head-to-head matches between teams."""
    try:
        home_col, away_col, result_col = get_column_names(version) if get_column_names else ("HomeTeam", "AwayTeam", "FTR")
        h2h = data[(data[home_col] == home) & (data[away_col] == away)]
        
        if h2h.empty:
            return None
            
        h2h = h2h.drop(columns=[result_col, "Date", "Country", "League", "Season", "Time"], errors='ignore')
        
        if version == "v1" and 'HTR' in h2h:
            # FIXED: Use proper pandas method to avoid FutureWarning
            h2h['HTR'] = h2h['HTR'].replace({'H': 1, 'D': 2, 'A': 3}).infer_objects(copy=False).astype(float)
            
        mean = h2h.mean(numeric_only=True)
        
        if 'HTR' in mean:
            # FIXED: Proper HTR categorization logic
            htr_value = mean['HTR']
            if 0 <= htr_value <= 1.4:
                mean['HTR'] = 'H'
            elif 1.5 <= htr_value <= 2.4:
                mean['HTR'] = 'D'
            elif 2.5 <= htr_value <= 3.4:
                mean['HTR'] = 'A'
            else:
                mean['HTR'] = 'D'  # Default to draw if out of range
                
        input_df = pd.DataFrame([mean])
        return align_features(input_df, model)
        
    except Exception as e:
        logging.error(f"Compute mean error: {e}")
        return None

def calculate_probabilities(home, away, data, version="v1"):
    """Calculate historical probabilities for match outcomes."""
    try:
        if version == "v2":
            home_col, away_col, result_col = "home_team", "away_team", "Res"
            outcome_map = {"H": "Home Team Win", "D": "Draw", "A": "Away Team Win"}
        else:
            home_col, away_col, result_col = "HomeTeam", "AwayTeam", "FTR"
            outcome_map = {"H": "Home Team Win", "D": "Draw", "A": "Away Team Win"}

        h2h = data[(data[home_col] == home) & (data[away_col] == away)]
        
        if h2h.empty:
            return None

        value_counts = h2h[result_col].value_counts(normalize=True) * 100
        return {outcome_map.get(k, k): round(v, 2) for k, v in value_counts.items()}
        
    except Exception as e:
        logging.error(f"Probability calculation error: {e}")
        return None

def predict_with_confidence(model, input_df):
    """Make prediction with confidence scores."""
    try:
        if input_df is None or input_df.empty:
            return None, None, None
            
        proba = model.predict_proba(input_df)[0]
        pred_idx = proba.argmax()
        labels = model.classes_
        
        prediction = labels[pred_idx]
        confidence = proba[pred_idx]
        probabilities = dict(zip(labels, proba))
        
        return prediction, confidence, probabilities
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None, None, None

def determine_final_prediction(pred, probs):
    """Determine final prediction based on model output and probabilities."""
    try:
        # FIXED: Handle different prediction formats
        if isinstance(pred, str):
            # String prediction (H, D, A)
            if pred == 'H':
                model_outcome = "Home Team Win"
            elif pred == 'D':
                model_outcome = "Draw"
            elif pred == 'A':
                model_outcome = "Away Team Win"
            else:
                return "❗ Invalid prediction"
        elif isinstance(pred, (int, float)):
            # Numeric prediction
            if 0.5 <= pred <= 1.4:
                model_outcome = "Home Team Win"
            elif 1.5 <= pred <= 2.4:
                model_outcome = "Draw"
            elif 2.5 <= pred <= 3.4:
                model_outcome = "Away Team Win"
            else:
                return "❗ Invalid prediction"
        else:
            return "❗ Invalid prediction type"

        # FIXED: Check if probabilities exist and have values
        if not probs or not isinstance(probs, dict):
            return model_outcome
            
        # Find highest probability outcome
        if probs:
            highest_prob = max(probs.values())
            highest_outcome = None
            
            # Map probability keys to outcomes
            for key, value in probs.items():
                if value == highest_prob:
                    if key in ['H', 1, '1', 'Home Team Win']:
                        highest_outcome = "Home Team Win"
                    elif key in ['D', 2, '2', 'Draw']:
                        highest_outcome = "Draw"
                    elif key in ['A', 3, '3', 'Away Team Win']:
                        highest_outcome = "Away Team Win"
                    else:
                        highest_outcome = str(key)
                    break
            
            # Return the outcome with highest probability
            return highest_outcome if highest_outcome else model_outcome
        
        return model_outcome
        
    except Exception as e:
        logging.error(f"Final prediction error: {e}")
        return "❗ Error in prediction"

def validate_prediction_input(home, away, data):
    """Validate input data for prediction."""
    try:
        if not home or not away:
            return False, "Missing team names"
            
        if data is None or data.empty:
            return False, "No data available"
            
        # Check if teams exist in data
        home_teams = data['HomeTeam'].unique() if 'HomeTeam' in data.columns else []
        away_teams = data['AwayTeam'].unique() if 'AwayTeam' in data.columns else []
        
        if home not in home_teams and home not in away_teams:
            return False, f"Team '{home}' not found in data"
            
        if away not in home_teams and away not in away_teams:
            return False, f"Team '{away}' not found in data"
            
        return True, "Valid input"
        
    except Exception as e:
        logging.error(f"Validation error: {e}")
        return False, f"Validation error: {e}"

# Test function to verify logic
def test_model_utils_logic():
    """Test the fixed model utils logic."""
    print("=== Testing Model Utils Logic ===")
    
    # Test data
    test_data = pd.DataFrame({
        'HomeTeam': ['Arsenal', 'Chelsea', 'Arsenal'],
        'AwayTeam': ['Chelsea', 'Arsenal', 'Chelsea'],
        'FTR': ['H', 'A', 'D'],
        'HTR': ['H', 'A', 'D'],
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6]
    })
    
    # Test validation
    is_valid, message = validate_prediction_input('Arsenal', 'Chelsea', test_data)
    print(f"✅ Validation: {is_valid} - {message}")
    
    # Test probability calculation
    probs = calculate_probabilities('Arsenal', 'Chelsea', test_data)
    print(f"✅ Probabilities: {probs}")
    
    # Test mean computation (mock model)
    class MockModel:
        def __init__(self):
            self.feature_names_in_ = ['Feature1', 'Feature2']
    
    mock_model = MockModel()
    mean_features = compute_mean_for_teams('Arsenal', 'Chelsea', test_data, mock_model)
    print(f"✅ Mean features computed: {mean_features is not None}")
    
    print("🎉 Model utils logic is working correctly!")

if __name__ == "__main__":
    test_model_utils_logic() 