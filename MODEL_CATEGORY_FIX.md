# Model Category Fix Implementation

## Overview
This document explains the changes made to implement different models for "Home" and "Away" categories in the football prediction app.

## Changes Made

### 1. Views.py Updates
- **predict() function**: Modified to use different models for home and away predictions
  - Home predictions use `model1.pkl`
  - Away predictions use `model2.pkl`
  - Added category parameters to track which categories were selected

- **prepare_features() function**: Enhanced to handle different feature sets for home and away predictions
  - Home features include home advantage flag and home team form
  - Away features include away team flag and away team form
  - Added `is_home` parameter to differentiate feature preparation

- **api_predict() function**: Updated to use different models and pass category information

### 2. Template Updates
- **predict.html**: Added validation to ensure categories are selected before prediction
- **result.html**: Updated to display which models were used for home and away predictions
  - Shows "Model 1" for home predictions
  - Shows "Model 2" for away predictions
  - Displays the selected categories

### 3. Model Usage
- **Home Team Predictions**: Uses `model1.pkl` with home-optimized features
- **Away Team Predictions**: Uses `model2.pkl` with away-optimized features

## Feature Differences

### Model Features (Both models use same 4 features)
- Home team strength (0-1)
- Away team strength (0-1)
- Combined strength
- Strength difference

**Note**: Both models use the same 4 features, but they are trained differently:
- **Model 1 (Home)**: Optimized for predicting home team performance
- **Model 2 (Away)**: Optimized for predicting away team performance

## Benefits
1. **Specialized Predictions**: Each model is optimized for its specific prediction type
2. **Better Accuracy**: Different feature sets for home vs away scenarios
3. **Transparency**: Users can see which models were used for their predictions
4. **Category Tracking**: System tracks which categories were selected for analysis

## Usage
1. Select categories for both home and away teams
2. Choose leagues and teams as before
3. Submit prediction
4. View results showing which models were used for each prediction type

## Files Modified
- `predictor/views.py`
- `templates/predictor/predict.html`
- `templates/predictor/result.html`
- `MODEL_CATEGORY_FIX.md` (this file) 