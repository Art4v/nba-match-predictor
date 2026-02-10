# nba_data_fetch.py
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import os 

def fetch_nba_games(start_season='2023-24', output_file='games.csv'):
    """
    Fetch NBA games from start_season onwards and save to CSV in the 'data/' folder.
    """

    # create save folder if it doesn't exist
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', output_file)

    print(f"Fetching NBA games for season {start_season}...")

    # initialize the LeagueGameFinder endpoint
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=start_season)

    # convert result to pandas DataFrame
    games_df = gamefinder.get_data_frames()[0]

    # keep only relevant columns
    columns_to_keep = [
        'GAME_ID',
        'GAME_DATE',
        'TEAM_ID',
        'TEAM_ABBREVIATION',
        'TEAM_NAME',
        'MATCHUP',
        'WL',
        'PTS',
        'REB',
        'AST'
    ]
    games_df = games_df[columns_to_keep]

    # convert 'WL' to numeric for ML (W=1, L=0)
    games_df['WL'] = games_df['WL'].map({'W': 1, 'L': 0})

    # save to CSV in the data folder
    games_df.to_csv(file_path, index=False)
    print(f"Saved {len(games_df)} games to {file_path}")

if __name__ == "__main__":
    fetch_nba_games()
