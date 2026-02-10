import pandas as pd
import os
from nba_api.stats.endpoints import (
    leaguegamefinder,
    leaguedashteamstats,
    leaguedashplayerstats,
)
from datetime import datetime
import time

class NBADataParser:
    """Extract data from NBA API and save as CSV files."""
    
    def __init__(self, output_dir='./data'):
        """
        Initialize the parser.
        
        Args:
            output_dir: Directory to save CSV files (default: 'data')
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def fetch_games(self, from_year=2023):
        """
        Fetch games data from specified year onwards and save to CSV.
        
        Args:
            from_year: Start fetching from this year onwards (default: 2023)
        """
        print(f"Fetching games data from {from_year} onwards...")
        try:
            games_finder = leaguegamefinder.LeagueGameFinder()
            games_df = games_finder.get_data_frames()[0]
            
            # Convert GAME_DATE to datetime
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            
            # Filter to include data from specified year onwards
            games_df = games_df[games_df['GAME_DATE'].dt.year >= from_year]
            games_df = games_df.sort_values('GAME_DATE')
            
            output_path = os.path.join(self.output_dir, 'games.csv')
            games_df.to_csv(output_path, index=False)
            year_range = f"{from_year}-{games_df['GAME_DATE'].dt.year.max()}"
            print(f"[OK] Games data saved: {output_path} ({len(games_df)} rows, {year_range})")
            return games_df
        except Exception as e:
            print(f"[ERROR] Error fetching games: {e}")
            return None
    
    def fetch_team_stats(self, from_year=2023):
        """
        Fetch team stats for multiple seasons from specified year onwards and save to CSV.
        
        Args:
            from_year: Start fetching from this year onwards (default: 2023)
        """
        print(f"Fetching team stats from {from_year} onwards...")
        try:
            all_teams_df = []
            current_year = datetime.now().year
            
            # Fetch stats for each season from from_year to current year
            for year in range(from_year, current_year + 1):
                season = f"{year}-{str(year + 1)[-2:]}"
                print(f"  Fetching {season}...", end=' ')
                try:
                    team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
                    teams_df = team_stats.get_data_frames()[0]
                    teams_df['SEASON'] = season
                    all_teams_df.append(teams_df)
                    print("[OK]")
                except Exception as e:
                    print(f"[SKIP] {e}")
                time.sleep(0.3)  # Rate limiting
            
            if all_teams_df:
                # Drop all-NA columns before concatenating to avoid FutureWarning
                for i, df in enumerate(all_teams_df):
                    all_teams_df[i] = df.dropna(axis=1, how='all')
                
                teams_df = pd.concat(all_teams_df, ignore_index=True, sort=False)
                output_path = os.path.join(self.output_dir, 'teams.csv')
                teams_df.to_csv(output_path, index=False)
                print(f"[OK] Team stats saved: {output_path} ({len(teams_df)} rows)")
                return teams_df
            else:
                print(f"[ERROR] No team stats fetched")
                return None
        except Exception as e:
            print(f"[ERROR] Error fetching team stats: {e}")
            return None
    
    def fetch_player_stats(self, from_year=2023):
        """
        Fetch player stats for multiple seasons from specified year onwards and save to CSV.
        
        Args:
            from_year: Start fetching from this year onwards (default: 2023)
        """
        print(f"Fetching player stats from {from_year} onwards...")
        try:
            all_players_df = []
            current_year = datetime.now().year
            
            # Fetch stats for each season from from_year to current year
            for year in range(from_year, current_year + 1):
                season = f"{year}-{str(year + 1)[-2:]}"
                print(f"  Fetching {season}...", end=' ')
                try:
                    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
                    players_df = player_stats.get_data_frames()[0]
                    players_df['SEASON'] = season
                    all_players_df.append(players_df)
                    print("[OK]")
                except Exception as e:
                    print(f"[SKIP] {e}")
                time.sleep(0.3)  # Rate limiting
            
            if all_players_df:
                # Drop all-NA columns before concatenating to avoid FutureWarning
                for i, df in enumerate(all_players_df):
                    all_players_df[i] = df.dropna(axis=1, how='all')
                
                players_df = pd.concat(all_players_df, ignore_index=True, sort=False)
                output_path = os.path.join(self.output_dir, 'players.csv')
                players_df.to_csv(output_path, index=False)
                print(f"[OK] Player stats saved: {output_path} ({len(players_df)} rows)")
                return players_df
            else:
                print(f"[ERROR] No player stats fetched")
                return None
        except Exception as e:
            print(f"[ERROR] Error fetching player stats: {e}")
            return None
    
    def fetch_all(self, from_year=2023):
        """
        Fetch all data from specified year onwards and save to CSV files.
        
        Args:
            from_year: Start fetching from this year onwards (default: 2023)
        """
        print(f"\n{'='*50}")
        print(f"NBA Data Parser - Fetching data from {from_year} onwards")
        print(f"{'='*50}\n")
        
        start_time = time.time()
        
        games_df = self.fetch_games(from_year=from_year)
        time.sleep(0.5)  # Rate limiting
        
        teams_df = self.fetch_team_stats(from_year=from_year)
        time.sleep(0.5)  # Rate limiting
        
        players_df = self.fetch_player_stats(from_year=from_year)
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Data extraction complete in {elapsed_time:.2f} seconds")
        print(f"Files saved to: {os.path.abspath(self.output_dir)}")
        print(f"{'='*50}\n")
        
        return {
            'games': games_df,
            'teams': teams_df,
            'players': players_df
        }


if __name__ == '__main__':
    # Example usage - fetch all data from 2023 onwards
    parser = NBADataParser()
    data = parser.fetch_all(from_year=2023)
