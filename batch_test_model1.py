import os
import sys
import joblib
import pandas as pd
from predictor.analytics import predict_with_confidence, compute_mean_for_teams, load_football_data, get_column_names

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
import django
django.setup()

# Team list for model1 (major European leagues: EPL, La Liga, Bundesliga, Serie A, Ligue 1)
# model2 is meant to train/predict for other leagues
TEAM_LIST = [
    'Manchester City', 'Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham Hotspur', 'Newcastle United', 'West Ham United',
    'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia', 'Villarreal',
    'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Borussia Monchengladbach',
    'Juventus', 'Inter Milan', 'AC Milan', 'Napoli', 'Roma', 'Lazio',
    'Paris Saint-Germain', 'Marseille', 'Lyon', 'Monaco', 'Nice'
]

# Load model1
model1_path = os.path.join(os.path.dirname(__file__), 'models', 'model1.pkl')
model1 = joblib.load(model1_path)

# Load match data
match_data = load_football_data()
if match_data is None:
    print("Could not load football data. Exiting.")
    sys.exit(1)

results = []
total_attempts = 0
skipped = 0

for home_team in TEAM_LIST:
    for away_team in TEAM_LIST:
        if home_team == away_team:
            continue
        total_attempts += 1
        features = compute_mean_for_teams(home_team, away_team, match_data, model1, get_column_names, "v1")
        if features is None:
            skipped += 1
            continue
        print(f"[DEBUG] Features for {home_team} vs {away_team}:\n{features.head()}")
        prediction, confidence, probabilities = predict_with_confidence(model1, features)
        print(f"Home: {home_team:22} | Away: {away_team:22} | Pred: {prediction} | Conf: {confidence:.2f} | Probs: {probabilities}")
        results.append(probabilities)

# Summarize how often model1 outputs 100% for a single class
one_hot_count = 0
for probs in results:
    if any(abs(v - 1.0) < 1e-6 for v in probs.values()):
        if sum(1 for v in probs.values() if abs(v - 1.0) < 1e-6) == 1 and sum(1 for v in probs.values() if abs(v) < 1e-6) == (len(probs)-1):
            one_hot_count += 1

total = len(results)
print(f"\nModel1 output 100% for a single class in {one_hot_count}/{total} matches ({one_hot_count/total*100:.1f}%)")
print(f"Skipped {skipped} matches due to lack of head-to-head data (out of {total_attempts} total attempts)") 