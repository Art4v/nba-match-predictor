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
    
    def __init__(self, output_dir='.'):
        """
        Initialize the parser.
        
        Args:
            output_dir: Directory to save CSV files (default: current directory)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def fetch_games(self, season='2023-24'):
        """
        Fetch games data and save to CSV.
        
        Args:
            season: Season to fetch (default: '2023-24')
        """
        print(f"Fetching games data for {season}...")
        try:
            games_finder = leaguegamefinder.LeagueGameFinder()
            games_df = games_finder.get_data_frames()[0]
            
            # Filter by season if needed
            if 'SEASON_ID' in games_df.columns:
                season_id = int(season.split('-')[0])
                games_df = games_df[games_df['SEASON_ID'] == season_id]
            
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values('GAME_DATE')
            
            output_path = os.path.join(self.output_dir, 'games.csv')
            games_df.to_csv(output_path, index=False)
            print(f"[OK] Games data saved: {output_path} ({len(games_df)} rows)")
            return games_df
        except Exception as e:
            print(f"[ERROR] Error fetching games: {e}")
            return None
    
    def fetch_team_stats(self, season='2023-24'):
        """
        Fetch team stats and save to CSV.
        
        Args:
            season: Season to fetch (default: '2023-24')
        """
        print(f"Fetching team stats for {season}...")
        try:
            team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
            teams_df = team_stats.get_data_frames()[0]
            
            output_path = os.path.join(self.output_dir, 'teams.csv')
            teams_df.to_csv(output_path, index=False)
            print(f"[OK] Team stats saved: {output_path} ({len(teams_df)} rows)")
            return teams_df
        except Exception as e:
            print(f"[ERROR] Error fetching team stats: {e}")
            return None
    
    def fetch_player_stats(self, season='2023-24'):
        """
        Fetch player stats and save to CSV.
        
        Args:
            season: Season to fetch (default: '2023-24')
        """
        print(f"Fetching player stats for {season}...")
        try:
            player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
            players_df = player_stats.get_data_frames()[0]
            
            output_path = os.path.join(self.output_dir, 'players.csv')
            players_df.to_csv(output_path, index=False)
            print(f"[OK] Player stats saved: {output_path} ({len(players_df)} rows)")
            return players_df
        except Exception as e:
            print(f"[ERROR] Error fetching player stats: {e}")
            return None
    
    def fetch_all(self, season='2023-24'):
        """
        Fetch all data and save to CSV files.
        
        Args:
            season: Season to fetch (default: '2023-24')
        """
        print(f"\n{'='*50}")
        print(f"NBA Data Parser - Fetching {season} Season")
        print(f"{'='*50}\n")
        
        start_time = time.time()
        
        games_df = self.fetch_games(season)
        time.sleep(0.5)  # Rate limiting
        
        teams_df = self.fetch_team_stats(season)
        time.sleep(0.5)  # Rate limiting
        
        players_df = self.fetch_player_stats(season)
        
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
    # Example usage
    parser = NBADataParser(output_dir='.')
    data = parser.fetch_all(season='2023-24')
