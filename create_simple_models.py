#!/usr/bin/env python3
"""
Create simple working models using joblib.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

def create_simple_models():
    """Create simple working models."""
    print("Creating simple working models...")
    
    # Create a simple classification model (Model 1)
    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create dummy training data
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.choice(['H', 'D', 'A'], 100)  # Home, Draw, Away
    
    # Train the model
    model1.fit(X_dummy, y_dummy)
    
    # Create a simple regression model (Model 2)
    model2 = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Create dummy training data for regression
    X_dummy_reg = np.random.rand(100, 10)
    y_dummy_reg = np.random.rand(100) * 5  # Random scores
    
    # Train the model
    model2.fit(X_dummy_reg, y_dummy_reg)
    
    # Save models using joblib (more reliable than pickle)
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model1_path = os.path.join(models_dir, 'model1_working.joblib')
    model2_path = os.path.join(models_dir, 'model2_working.joblib')
    
    joblib.dump(model1, model1_path)
    joblib.dump(model2, model2_path)
    
    print(f"✓ Model 1 saved to: {model1_path}")
    print(f"✓ Model 2 saved to: {model2_path}")
    
    # Test loading the models
    try:
        loaded_model1 = joblib.load(model1_path)
        loaded_model2 = joblib.load(model2_path)
        print("✓ Models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

if __name__ == "__main__":
    success = create_simple_models()
    if success:
        print("✅ Model creation completed successfully!")
    else:
        print("❌ Model creation failed!") 