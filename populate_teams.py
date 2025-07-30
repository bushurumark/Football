#!/usr/bin/env python
"""
Script to populate the database with sample teams for testing.
Run this after setting up the database.
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'football_predictor.settings')
django.setup()

from predictor.models import Team

def populate_teams():
    """Populate the database with sample teams."""
    
    teams_data = [
        # Premier League Teams
        {'name': 'Manchester City', 'league': 'Premier League', 'country': 'England'},
        {'name': 'Manchester United', 'league': 'Premier League', 'country': 'England'},
        {'name': 'Liverpool', 'league': 'Premier League', 'country': 'England'},
        {'name': 'Chelsea', 'league': 'Premier League', 'country': 'England'},
        {'name': 'Arsenal', 'league': 'Premier League', 'country': 'England'},
        {'name': 'Tottenham Hotspur', 'league': 'Premier League', 'country': 'England'},
        {'name': 'Newcastle United', 'league': 'Premier League', 'country': 'England'},
        {'name': 'West Ham United', 'league': 'Premier League', 'country': 'England'},
        
        # La Liga Teams
        {'name': 'Real Madrid', 'league': 'La Liga', 'country': 'Spain'},
        {'name': 'Barcelona', 'league': 'La Liga', 'country': 'Spain'},
        {'name': 'Atletico Madrid', 'league': 'La Liga', 'country': 'Spain'},
        {'name': 'Sevilla', 'league': 'La Liga', 'country': 'Spain'},
        {'name': 'Valencia', 'league': 'La Liga', 'country': 'Spain'},
        {'name': 'Villarreal', 'league': 'La Liga', 'country': 'Spain'},
        
        # Bundesliga Teams
        {'name': 'Bayern Munich', 'league': 'Bundesliga', 'country': 'Germany'},
        {'name': 'Borussia Dortmund', 'league': 'Bundesliga', 'country': 'Germany'},
        {'name': 'RB Leipzig', 'league': 'Bundesliga', 'country': 'Germany'},
        {'name': 'Bayer Leverkusen', 'league': 'Bundesliga', 'country': 'Germany'},
        {'name': 'Borussia Monchengladbach', 'league': 'Bundesliga', 'country': 'Germany'},
        
        # Serie A Teams
        {'name': 'Juventus', 'league': 'Serie A', 'country': 'Italy'},
        {'name': 'Inter Milan', 'league': 'Serie A', 'country': 'Italy'},
        {'name': 'AC Milan', 'league': 'Serie A', 'country': 'Italy'},
        {'name': 'Napoli', 'league': 'Serie A', 'country': 'Italy'},
        {'name': 'Roma', 'league': 'Serie A', 'country': 'Italy'},
        {'name': 'Lazio', 'league': 'Serie A', 'country': 'Italy'},
        
        # Ligue 1 Teams
        {'name': 'Paris Saint-Germain', 'league': 'Ligue 1', 'country': 'France'},
        {'name': 'Marseille', 'league': 'Ligue 1', 'country': 'France'},
        {'name': 'Lyon', 'league': 'Ligue 1', 'country': 'France'},
        {'name': 'Monaco', 'league': 'Ligue 1', 'country': 'France'},
        {'name': 'Nice', 'league': 'Ligue 1', 'country': 'France'},
    ]
    
    created_count = 0
    for team_data in teams_data:
        team, created = Team.objects.get_or_create(
            name=team_data['name'],
            defaults={
                'league': team_data['league'],
                'country': team_data['country']
            }
        )
        if created:
            created_count += 1
            print(f"Created team: {team.name}")
    
    print(f"\nTotal teams created: {created_count}")
    print(f"Total teams in database: {Team.objects.count()}")

if __name__ == '__main__':
    print("Populating teams database...")
    populate_teams()
    print("Done!") 