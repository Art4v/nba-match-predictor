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

    # convert GAME_DATE to datetime safely
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'], errors='coerce')

    # extract numeric date features
    games_df['GAME_MONTH'] = games_df['GAME_DATE'].dt.month
    games_df['GAME_DAY'] = games_df['GAME_DATE'].dt.day
    games_df['GAME_WEEKDAY'] = games_df['GAME_DATE'].dt.weekday  # 0 = Monday

    # encode home/away from MATCHUP
    games_df['IS_HOME'] = games_df['MATCHUP'].str.contains(' vs ').astype(int)  # 1=home, 0=away

    # convert WL to numeric for ML
    games_df['WL'] = games_df['WL'].map({'W': 1, 'L': 0})

    # drop original string/date columns that are no longer needed for ML
    games_df = games_df.drop(columns=['GAME_DATE', 'MATCHUP', 'TEAM_NAME', 'TEAM_ABBREVIATION'])

    # save to CSV in the data folder
    games_df.to_csv(file_path, index=False)
    print(f"Saved {len(games_df)} games to {file_path}")

if __name__ == "__main__":
    fetch_nba_games()
