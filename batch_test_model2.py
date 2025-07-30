import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()
import joblib
import pandas as pd
from predictor.analytics import compute_mean_for_teams, load_football_data, get_column_names

TEAM_LIST = [
    'Manchester City', 'Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham Hotspur', 'Newcastle United', 'West Ham United',
    'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia', 'Villarreal',
    'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Borussia Monchengladbach',
    'Juventus', 'Inter Milan', 'AC Milan', 'Napoli', 'Roma', 'Lazio',
    'Paris Saint-Germain', 'Marseille', 'Lyon', 'Monaco', 'Nice'
]

# Load model2
model2_path = os.path.join(os.path.dirname(__file__), 'models', 'model2.pkl')
model2 = joblib.load(model2_path)

# Load match data
match_data = load_football_data()
if match_data is None:
    print("Could not load football data. Exiting.")
    import sys
    sys.exit(1)

results = []
total_attempts = 0
skipped = 0
class_counts = {1: 0, 2: 0, 3: 0}

for home_team in TEAM_LIST:
    for away_team in TEAM_LIST:
        if home_team == away_team:
            continue
        total_attempts += 1
        # Try to compute features using H2H or team averages; if not enough data, features will be None
        features = compute_mean_for_teams(home_team, away_team, match_data, model2, get_column_names, "v1")
        if features is None:
            skipped += 1
            continue
        print(f"[DEBUG] Features for {home_team} vs {away_team}:\n{features.head()}")
        # Model 2 is a regressor
        try:
            reg_pred = model2.predict(features)[0]
        except Exception as e:
            print(f"Error predicting for {home_team} vs {away_team}: {e}")
            continue
        # Convert regression output to class
        if reg_pred < 1.5:
            pred_class = 1  # Home win
        elif reg_pred < 2.5:
            pred_class = 2  # Draw
        else:
            pred_class = 3  # Away win
        class_counts[pred_class] += 1
        print(f"Home: {home_team:22} | Away: {away_team:22} | RegPred: {reg_pred:.2f} | Class: {pred_class}")
        results.append(pred_class)

print(f"\nClass distribution: Home win (1): {class_counts[1]}, Draw (2): {class_counts[2]}, Away win (3): {class_counts[3]}")
print(f"Skipped {skipped} matches due to lack of head-to-head or team data (out of {total_attempts} total attempts)") 