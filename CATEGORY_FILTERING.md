# Category-Based League Filtering

## Overview

The Football Prediction App now includes a category-based filtering system that organizes leagues into two main categories:

1. **European Leagues** - Major European football leagues
2. **Others** - Non-European leagues and other competitions

## How It Works

### User Interface Flow

1. **Select Category**: Users first choose between "European Leagues" or "Others"
2. **Select League**: Based on the category, available leagues are filtered and displayed
3. **Select Team**: Teams are filtered based on the selected league

### Categories and Leagues

#### European Leagues
- Premier League
- English Championship
- Serie A
- Serie B
- Ligue1
- Ligue2
- La Liga
- La Liga2
- Eredivisie
- Bundesliga
- Bundesliga2
- Scottish League
- Belgium League
- Portuguese League
- Turkish League
- Greece League

#### Others
- Switzerland League
- Denmark League
- Austria League
- Mexico League
- Russia League
- Romania League

## Technical Implementation

### Backend (Django)

1. **Data Structure**: Leagues are organized in a nested dictionary structure in `views.py`
2. **API Endpoint**: `/api/teams-by-category/` provides teams filtered by category and league
3. **Template Context**: League data is serialized to JSON for frontend use

### Frontend (JavaScript)

1. **Dynamic Filtering**: JavaScript handles the cascading dropdown updates
2. **Event Listeners**: Category changes trigger league updates, league changes trigger team updates
3. **Data Binding**: Uses the serialized league data from Django context

## Files Modified

- `predictor/views.py` - Added category data and API endpoint
- `predictor/urls.py` - Added new API route
- `templates/predictor/predict.html` - Updated UI for category selection
- `static/js/app.js` - Added category-based filtering functions

## Usage Example

1. User selects "European Leagues" category
2. League dropdown populates with European leagues
3. User selects "Premier League"
4. Team dropdown populates with Premier League teams
5. User selects home and away teams
6. Prediction is generated

## Benefits

- **Better Organization**: Leagues are logically grouped
- **Improved UX**: Users can quickly find relevant leagues
- **Scalability**: Easy to add new categories and leagues
- **Performance**: Reduced dropdown options improve loading times 