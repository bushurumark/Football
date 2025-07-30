from django.db import models
from django.contrib.auth.models import User


class Prediction(models.Model):
    """Model for storing football match predictions."""
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    home_score = models.IntegerField()
    away_score = models.IntegerField()
    prediction_date = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField()
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    def __str__(self):
        return f"{self.home_team} vs {self.away_team} - {self.home_score}:{self.away_score}"
    
    class Meta:
        ordering = ['-prediction_date']


class Match(models.Model):
    """Model for storing match data."""
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    home_score = models.IntegerField(null=True, blank=True)
    away_score = models.IntegerField(null=True, blank=True)
    match_date = models.DateField()
    league = models.CharField(max_length=100)
    season = models.CharField(max_length=20)
    
    def __str__(self):
        return f"{self.home_team} vs {self.away_team} ({self.league})"
    
    class Meta:
        ordering = ['-match_date']


class Team(models.Model):
    """Model for storing team information."""
    name = models.CharField(max_length=100, unique=True)
    league = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    
    def __str__(self):
        return f"{self.name} ({self.league})"
    
    class Meta:
        ordering = ['name']
