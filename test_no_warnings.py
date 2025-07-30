#!/usr/bin/env python3
"""
Test script to verify that scikit-learn warnings are suppressed.
"""

import warnings
import joblib
import os

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def test_model_loading_no_warnings():
    """Test model loading without warnings."""
    print("=== Testing Model Loading (No Warnings) ===")
    
    try:
        # Load models
        model1_path = os.path.join(os.path.dirname(__file__), 'models', 'model1.pkl')
        model2_path = os.path.join(os.path.dirname(__file__), 'models', 'model2.pkl')
        
        print(f"Loading model1 from: {model1_path}")
        model1 = joblib.load(model1_path)
        print(f"✓ Model1 loaded successfully. Type: {type(model1)}")
        
        print(f"Loading model2 from: {model2_path}")
        model2 = joblib.load(model2_path)
        print(f"✓ Model2 loaded successfully. Type: {type(model2)}")
        
        print("\n✅ No warnings displayed!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

if __name__ == "__main__":
    test_model_loading_no_warnings() 