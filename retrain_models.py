#!/usr/bin/env python3
"""
Script to retrain models with current scikit-learn version.
This eliminates version compatibility warnings.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

from predictor.analytics import load_football_data

def prepare_training_data():
    """Prepare data for training."""
    print("Loading football data...")
    data = load_football_data()
    
    if data is None:
        print("❌ Failed to load data")
        return None, None
    
    print(f"Data shape: {data.shape}")
    print(f"Unique matches: {data[['HomeTeam','AwayTeam']].drop_duplicates().shape[0]}")
    print(f"Unique teams: {len(set(data['HomeTeam']).union(set(data['AwayTeam'])))}")
    print(f"Matches per team (Home):\n{data['HomeTeam'].value_counts().head()}\n...")
    print(f"Matches per team (Away):\n{data['AwayTeam'].value_counts().head()}\n...")
    
    # Prepare features for model1 (classification)
    print("Preparing features for classification model...")
    
    # Create one-hot encoded features for teams
    home_teams = pd.get_dummies(data['HomeTeam'], prefix='HomeTeam')
    away_teams = pd.get_dummies(data['AwayTeam'], prefix='AwayTeam')
    
    # Combine all features
    feature_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 
                      'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 
                      'B365D', 'B365A', 'MaxD', 'MaxA', 'AvgH', 'B365<2.5', 
                      'Max>2.5', 'Max<2.5', 'B365AHH', 'B365AHA', 'MaxAHH', 
                      'MaxAHA', 'B365CD', 'B365CA', 'MaxCD', 'MaxCA', 'AvgCH', 
                      'B365C<2.5', 'MaxC>2.5', 'MaxC<2.5', 'AvgC<2.5', 
                      'B365CAHH', 'B365CAHA', 'MaxCAHH', 'MaxCAHA']
    
    # Select numeric features
    numeric_features = data[feature_columns].fillna(0)
    # Drop low-variance features
    low_var = numeric_features.var() < 0.01
    if low_var.any():
        print(f"Dropping low-variance features: {list(numeric_features.columns[low_var])}")
        numeric_features = numeric_features.loc[:, ~low_var]
    print("\nFeature statistics:")
    print(numeric_features.describe().T[['mean','std','min','max']])
    print("\nFeature variance (lowest 10):")
    print(numeric_features.var().sort_values().head(10))
    
    # Combine all features
    X = pd.concat([numeric_features, home_teams, away_teams], axis=1)
    
    # Prepare target for classification (FTR: H=Home, D=Draw, A=Away)
    y_class = data['FTR'].fillna('D')  # Fill missing with Draw
    print("\nClassification target distribution:")
    print(y_class.value_counts())
    
    # Prepare target for regression (total goals)
    y_reg = data['FTHG'].fillna(0) + data['FTAG'].fillna(0)
    print("\nRegression target (total goals) stats:")
    print(y_reg.describe())
    
    print(f"Features shape: {X.shape}")
    print(f"Classification target shape: {y_class.shape}")
    print(f"Regression target shape: {y_reg.shape}")
    
    return X, y_class, y_reg

def train_models():
    """Train new models with current scikit-learn version."""
    print("=== Training New Models ===")
    
    # Prepare data
    X, y_class, y_reg = prepare_training_data()
    
    if X is None:
        return False
    
    try:
        # Train classification model (Model1)
        print("\nTraining classification model...")
        model1 = DecisionTreeClassifier(random_state=42, max_depth=10)
        model1.fit(X, y_class)
        print("✓ Classification model trained successfully")
        
        # Train regression model (Model2)
        print("\nTraining regression model...")
        model2 = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model2.fit(X, y_reg)
        print("✓ Regression model trained successfully")
        
        # Save models
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model1_path = os.path.join(models_dir, 'model1_new.pkl')
        model2_path = os.path.join(models_dir, 'model2_new.pkl')
        
        joblib.dump(model1, model1_path)
        joblib.dump(model2, model2_path)
        
        print(f"✓ Models saved to:")
        print(f"  - {model1_path}")
        print(f"  - {model2_path}")
        
        # Test the new models
        print("\n=== Testing New Models ===")
        
        # Test classification
        sample_pred = model1.predict(X.head(1))
        print(f"Classification prediction: {sample_pred[0]}")
        
        # Test regression
        sample_pred_reg = model2.predict(X.head(1))
        print(f"Regression prediction: {sample_pred_reg[0]:.2f} total goals")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return False

def main():
    """Main function."""
    print("Football Model Retraining")
    print("=" * 40)
    
    success = train_models()
    
    if success:
        print("\n✅ Models retrained successfully!")
        print("You can now replace the old models with the new ones:")
        print("1. Rename model1_new.pkl to model1.pkl")
        print("2. Rename model2_new.pkl to model2.pkl")
        print("3. Restart your Django server")
    else:
        print("\n❌ Model retraining failed!")

if __name__ == "__main__":
    main() 