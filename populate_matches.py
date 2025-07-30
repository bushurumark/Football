import os
import django
import random
from datetime import date, timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
django.setup()

from predictor.models import Match, Team

# Get all teams
teams = list(Team.objects.all())
leagues = list(set(team.league for team in teams))
seasons = ['2023/2024', '2024/2025']

sample_matches = []
for i in range(10):
    home_team = random.choice(teams)
    # Ensure away team is not the same as home team
    away_team = random.choice([t for t in teams if t != home_team])
    match_date = date.today() + timedelta(days=i)
    league = home_team.league
    season = random.choice(seasons)
    match = Match(
        home_team=home_team.name,
        away_team=away_team.name,
        home_score=None,
        away_score=None,
        match_date=match_date,
        league=league,
        season=season
    )
    sample_matches.append(match)

Match.objects.bulk_create(sample_matches)
print(f"Added {len(sample_matches)} sample matches.") 