import json
import pandas as pd
import pickle
from datetime import datetime
import pytz
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load your Scikit-learn model here
with open('ml/team_independent_tm_normalized_teams_stats.sav','rb') as f:
    model = pickle.load(f)

with open('data/teams.json','rb') as f:
    teams_data = json.load(f)

teams_mapping = {
    esports_team["team_id"]: esports_team for esports_team in teams_data
}

class EloRatingSystem:
    def __init__(self):
        self.k_factor = 0.3094478*10
        self.ratings = {}

    def add_player(self, player_name, rating=1200):
        if player_name not in self.ratings:
            self.ratings[player_name] = rating

    def get_rating(self, player_name):
        return self.ratings.get(player_name, 1200)
    
    def get_all_ratings(self):
        return sorted(self.ratings.items(), key=lambda x:x[1], reverse=True)

    def calculate_expected_score(self, player_a, player_b, anon_a, anon_b):
        rating_a = self.get_rating(player_a) if not anon_a else 1200
        rating_b = self.get_rating(player_b) if not anon_b else 1200
        expected_score_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return expected_score_a
    
    def calculate_tournament_value(self,tournament_value,days_since_match,weight=10):
        return weight*tournament_value/days_since_match

    def update_ratings(self, 
                        winner, 
                        loser, 
                        performance_factor_winner, 
                        performance_factor_loser, 
                        tournament_value,
                        days_since_match,
                        anon_winner,
                        anon_loser):
        expected_score_winner = self.calculate_expected_score(winner, loser, anon_winner, anon_loser)
        expected_score_loser = 1 - expected_score_winner

        rating_winner = self.get_rating(winner)
        rating_loser = self.get_rating(loser)

        time_importance_delay = self.calculate_tournament_value(tournament_value,days_since_match)

        new_rating_winner = rating_winner + self.k_factor * time_importance_delay * performance_factor_winner * (1 - expected_score_winner)
        new_rating_loser = rating_loser + self.k_factor * time_importance_delay * performance_factor_loser * (0 - expected_score_loser)

        if not anon_winner:
            self.ratings[winner] = new_rating_winner
        if not anon_loser:
            self.ratings[loser] = new_rating_loser

class PerformanceCalculator:
    
    def __init__(self,team_model,player_model=None):
        self.team_performance_calculator = team_model
        self.player_performance_calculator = player_model
        
    def calculate_performance(self,match_data,team_perf_weight,player_perf_weight):
        teams_performances_data = match_data['team_stats']
        players_performances_data = match_data['players_stats']
        winner = match_data['winner']
        teams_probabilities = self.team_performance_calculator.predict_proba([teams_performances_data])[0]
        players_probabilities = self.player_performance_calculator.predict_proba([players_performances_data])[0]
        weighted_teams = teams_probabilities[winner] ** 3
        weighted_players = players_probabilities[winner] ** 3
        winning_team_performance = team_perf_weight*weighted_teams #+ player_perf_weight*weighted_players
        return winning_team_performance
    
    def calculate_team_performance(self,winner_data,loser_data):
        winner_chance = self.team_performance_calculator.predict_proba([winner_data])[0][1]
        loser_chance = self.team_performance_calculator.predict_proba([loser_data])[0][1]
        winner_performance = winner_chance*3.58049065
        loser_performance = loser_chance*3.44078367
        return winner_performance,loser_performance
        
    def calculate_players_performance(self,winner_data,loser_data,winner_performance_factor,loser_performance_factor):
        winner_chance = self.player_performance_calculator.predict_proba([winner_data])[0][1]
        loser_chance = self.player_performance_calculator.predict_proba([loser_data])[0][1]
        winner_performance = winner_chance*winner_performance_factor
        loser_performance = loser_chance*loser_performance_factor
        return winner_performance,loser_performance
    
    def calculate_performance_classic(self,teams_performances_data,winner,weights):
        if winner == 0:
            winner_data = teams_performances_data[::2]
            loser_data = teams_performances_data[1::2]
        else:
            winner_data = teams_performances_data[1::2]
            loser_data = teams_performances_data[::2]
        performance_diff = [np.sqrt((x - y)**2) for x, y in zip(winner_data, loser_data)]
        return np.dot(performance_diff,weights)
    
performance_calculator = PerformanceCalculator(team_model=model)

tournaments_values = {
    'academy':60,
    'major':100,
    'offseason':40,
    'other':30,
    'regional':85,
    'unknown':1
}

@app.get("/tournament_rankings/{tournament_id}")
async def get_tournament_rankings(tournament_id:str,stage:str=None):    
    if tournament_id != None:
        try:
            team_stats = pd.read_csv('data/normalized_tournament_team_stats.csv', sep=';', dtype={'teamOnlineID': str})
            # player_stats = pd.read_csv(f'{BUCKET}/{PLAYERS_KEY}.csv', sep=';')

            team_stats['matchTime'] = pd.to_datetime(team_stats['matchTime'])
            # # player_stats['matchTime'] = pd.to_datetime(player_stats['matchTime'])

            tournament_games_team_stats = team_stats[team_stats['tournamentID'].astype(str)==str(tournament_id)]
            # # # tournament_games_player_stats = player_stats[player_stats['tournamentID']==tournament_id]

            if stage != None:
                tournament_games_team_stats = tournament_games_team_stats[tournament_games_team_stats['stageName']==stage]
                # # # tournament_games_player_stats = tournament_games_player_stats[tournament_games_player_stats['tournamentID']==stage]

            tournament_teams = tournament_games_team_stats['teamOnlineID'].unique()

            end_time = tournament_games_team_stats['matchTime'].max()
            six_months_ago = end_time - pd.DateOffset(months=6)
            games_of_teams_in_tournament = team_stats[team_stats['teamOnlineID'].isin(tournament_teams)]['esportsGameID'].values

            considered_games_team_stats = team_stats[(team_stats['matchTime'] >= six_months_ago)&(team_stats['matchTime'] <= end_time)&(team_stats['esportsGameID'].isin(games_of_teams_in_tournament))].sort_values(by=['matchTime','platformGameID','teamID']).reset_index(drop=True)
            # # # # considered_games_player_stats = player_stats[(player_stats['matchTime'] >= six_months_ago)&(player_stats['matchTime'] <= end_time)&(player_stats['teamOnlineID'].isin(tournament_teams))].sort_values(by=['matchTime','platformGameID','teamID']).reset_index(drop=True)

            considered_games_team_stats['tournamentValue'] = considered_games_team_stats['tournamentCategory'].apply(lambda x: tournaments_values[x])
            current_date = pd.Timestamp.now(tz='UTC')
            considered_games_team_stats['DaysSince'] = (current_date - considered_games_team_stats['matchTime']).dt.days

            team_stats_values = considered_games_team_stats.drop(columns=[
                'platformGameID','teamOnlineID', 'tournamentID', 'tournamentName',
                'tournamentSlug', 'tournamentCategory', 'tournamentStartDate',
                'tournamentEndDate', 'stageName', 'sectionName', 'matchID','winningTeam',
                'esportsGameID', 'teamID', 'matchTime','DaysSince','tournamentValue']).fillna(0).values
            
            # player_stats_values = considered_games_player_stats.drop(columns=[
            #     'tournamentID', 'tournamentName', 'tournamentSlug',
            #     'tournamentCategory', 'tournamentStartDate', 'tournamentEndDate',
            #     'stageName', 'sectionName', 'matchID', 'esportsGameID',
            #     'matchTime','platformGameID',
            #     'teamOnlineID', 'playerOnlineID_1', 'playerOnlineID_2','winningTeam',
            #     'playerOnlineID_3', 'playerOnlineID_4', 'playerOnlineID_5','teamID']).fillna(0).values

            records = considered_games_team_stats.to_dict('records')

            # Iterate through the list in steps of two
            elo = EloRatingSystem()
            y_pred = []
            for i in range(0, len(records), 2):
                if i + 1 < len(records):
                    team100 = records[i]
                    team200 = records[i + 1]
                    days_since_match = team100['DaysSince']
                    anon_team100 = not team100['teamOnlineID'] in tournament_teams
                    anon_team200 = not team200['teamOnlineID'] in tournament_teams
                    tournament_value = team100['tournamentValue']
                    winner = team100['teamOnlineID'] if team100['winningTeam'] == 1 else team200['teamOnlineID']
                    loser = team100['teamOnlineID'] if team100['winningTeam'] == 0 else team200['teamOnlineID']
                    winner_data = team_stats_values[i] if team100['winningTeam'] == 1 else team_stats_values[i + 1]
                    loser_data = team_stats_values[i] if team100['winningTeam'] == 0 else team_stats_values[i + 1]
                    # winner_players_data = player_stats_values[i] if team100['winningTeam'] == 1 else player_stats_values[i + 1]
                    # loser_players_data = player_stats_values[i] if team100['winningTeam'] == 0 else player_stats_values[i + 1]
                    anon_winner = anon_team100 if team100['winningTeam'] == 1 else anon_team200
                    anon_loser = anon_team100 if team100['winningTeam'] == 0 else anon_team200
                    if not anon_team100:
                        elo.add_player(team100['teamOnlineID'])
                    if not anon_team200:
                        elo.add_player(team200['teamOnlineID'])
                    if not anon_team100 and not anon_team200:
                        expected_score = elo.calculate_expected_score(team100['teamOnlineID'], team200['teamOnlineID'], False, False)
                        if expected_score != 0.5:
                            predicted = (team100['winningTeam'] == 1 and expected_score > 0.5) or (team100['winningTeam'] == 0 and expected_score < 0.5)
                            y_pred.append(predicted)
                    performance_factor_winner,performance_factor_loser = performance_calculator.calculate_team_performance(winner_data,loser_data)
                    # performance_winner_players,performance_loser_players = performance_calculator.calculate_players_performance(winner_players_data,loser_players_data)
                    elo.update_ratings(
                        winner, 
                        loser, 
                        performance_factor_winner, 
                        performance_factor_loser, 
                        tournament_value,
                        days_since_match,
                        anon_winner,
                        anon_loser
                    )
            ratings = elo.get_all_ratings()

            rank = 0
            last_score = -1
            last_unknown = 1
            ranking = []
            for team_id,score in ratings:
                if str(team_id) in teams_mapping:
                    team_esports_data = teams_mapping[str(team_id)] 
                else: 
                    team_esports_data = {
                        'name':'unknown_'+ str(last_unknown),
                        'acronym': 'UNK' + str(last_unknown)
                        }
                    last_unknown += 1
                if int(score) != int(last_score):
                    rank += 1
                ranking.append({
                    'team_id': team_id, 
                    'team_code': team_esports_data['acronym'], 
                    'team_name': team_esports_data['name'], 
                    'rank': rank, 
                })
            # Return the result as a JSON response
            response = {
                'statusCode': 200,
                'body': {'result': ranking}
            }
        except Exception as e:
            response = {
                'statusCode': 400,
                'body': {'error': str(e)}
            }
    else:
        # If 'a' or 'b' is missing, return an error response
        response = {
            'statusCode': 400,
            'body': {'error': 'Missing parameter "tournament_id"'}
        }
    return response

@app.get("/global_rankings/")
async def get_global_rankings(number_of_teams:int=20): 
    try:
        team_stats = pd.read_csv('data/normalized_tournament_team_stats.csv', sep=';', dtype={'teamOnlineID': str})
        # player_stats = pd.read_csv(f'{BUCKET}/{PLAYERS_KEY}.csv', sep=';')

        team_stats['matchTime'] = pd.to_datetime(team_stats['matchTime'])
        # # player_stats['matchTime'] = pd.to_datetime(player_stats['matchTime'])

        team_stats['tournamentValue'] = team_stats['tournamentCategory'].apply(lambda x: tournaments_values[x])
        current_date = pd.Timestamp.now(tz='UTC')
        team_stats['DaysSince'] = (current_date - team_stats['matchTime']).dt.days

        team_stats_values = team_stats.drop(columns=[
            'platformGameID','teamOnlineID', 'tournamentID', 'tournamentName',
            'tournamentSlug', 'tournamentCategory', 'tournamentStartDate',
            'tournamentEndDate', 'stageName', 'sectionName', 'matchID','winningTeam',
            'esportsGameID', 'teamID', 'matchTime','DaysSince','tournamentValue']).fillna(0).values
        
        # player_stats_values = considered_games_player_stats.drop(columns=[
        #     'tournamentID', 'tournamentName', 'tournamentSlug',
        #     'tournamentCategory', 'tournamentStartDate', 'tournamentEndDate',
        #     'stageName', 'sectionName', 'matchID', 'esportsGameID',
        #     'matchTime','platformGameID',
        #     'teamOnlineID', 'playerOnlineID_1', 'playerOnlineID_2','winningTeam',
        #     'playerOnlineID_3', 'playerOnlineID_4', 'playerOnlineID_5','teamID']).fillna(0).values

        records = team_stats.to_dict('records')

        # Iterate through the list in steps of two
        elo = EloRatingSystem()
        y_pred = []
        for i in range(0, len(records), 2):
            if i + 1 < len(records):
                team100 = records[i]
                team200 = records[i + 1]
                days_since_match = team100['DaysSince']
                tournament_value = team100['tournamentValue']
                winner = team100['teamOnlineID'] if team100['winningTeam'] == 1 else team200['teamOnlineID']
                loser = team100['teamOnlineID'] if team100['winningTeam'] == 0 else team200['teamOnlineID']
                winner_data = team_stats_values[i] if team100['winningTeam'] == 1 else team_stats_values[i + 1]
                loser_data = team_stats_values[i] if team100['winningTeam'] == 0 else team_stats_values[i + 1]
                # winner_players_data = player_stats_values[i] if team100['winningTeam'] == 1 else player_stats_values[i + 1]
                # loser_players_data = player_stats_values[i] if team100['winningTeam'] == 0 else player_stats_values[i + 1]
                elo.add_player(team100['teamOnlineID'])
                elo.add_player(team200['teamOnlineID'])
                expected_score = elo.calculate_expected_score(team100['teamOnlineID'], team200['teamOnlineID'], False, False)
                if expected_score != 0.5:
                    predicted = (team100['winningTeam'] == 1 and expected_score > 0.5) or (team100['winningTeam'] == 0 and expected_score < 0.5)
                    y_pred.append(predicted)
                performance_factor_winner,performance_factor_loser = performance_calculator.calculate_team_performance(winner_data,loser_data)
                # performance_winner_players,performance_loser_players = performance_calculator.calculate_players_performance(winner_players_data,loser_players_data)
                elo.update_ratings(
                    winner, 
                    loser, 
                    performance_factor_winner, 
                    performance_factor_loser, 
                    tournament_value,
                    days_since_match,
                    False,
                    False
                )
        ratings = elo.get_all_ratings()[:number_of_teams]

        rank = 0
        last_score = -1
        last_unknown = 1
        ranking = []
        for team_id,score in ratings:
            if str(team_id) in teams_mapping:
                team_esports_data = teams_mapping[str(team_id)] 
            else: 
                team_esports_data = {
                    'name':'unknown_'+ str(last_unknown),
                    'acronym': 'UNK' + str(last_unknown)
                    }
                last_unknown += 1
            if int(score) != int(last_score):
                rank += 1
            ranking.append({
                'team_id': team_id, 
                'team_code': team_esports_data['acronym'], 
                'team_name': team_esports_data['name'], 
                'rank': rank, 
            })
        # Return the result as a JSON response
        response = {
            'statusCode': 200,
            'body': {'result': ranking}
        }
    except Exception as e:
        response = {
            'statusCode': 400,
            'body': {'error': str(e)}
        }
    return response

class newList(BaseModel):
  team_ids: List[str]

@app.get("/team_rankings/")
async def get_team_rankings(team_ids: list = Query([])):
    if team_ids != None:
        try:
            team_stats = pd.read_csv('data/normalized_tournament_team_stats.csv', sep=';', dtype={'teamOnlineID': str})
            team_stats['matchTime'] = pd.to_datetime(team_stats['matchTime'])
            team_stats['teamOnlineID'] = team_stats['teamOnlineID'].astype(str)

            team_stats['tournamentValue'] = team_stats['tournamentCategory'].apply(lambda x: tournaments_values[x])
            current_date = pd.Timestamp.now(tz='UTC')
            team_stats['DaysSince'] = (current_date - team_stats['matchTime']).dt.days

            team_stats_values = team_stats.drop(columns=[
                'platformGameID','teamOnlineID', 'tournamentID', 'tournamentName',
                'tournamentSlug', 'tournamentCategory', 'tournamentStartDate',
                'tournamentEndDate', 'stageName', 'sectionName', 'matchID','winningTeam',
                'esportsGameID', 'teamID', 'matchTime','DaysSince','tournamentValue']).fillna(0).values
            
            # player_stats_values = considered_games_player_stats.drop(columns=[
            #     'tournamentID', 'tournamentName', 'tournamentSlug',
            #     'tournamentCategory', 'tournamentStartDate', 'tournamentEndDate',
            #     'stageName', 'sectionName', 'matchID', 'esportsGameID',
            #     'matchTime','platformGameID',
            #     'teamOnlineID', 'playerOnlineID_1', 'playerOnlineID_2','winningTeam',
            #     'playerOnlineID_3', 'playerOnlineID_4', 'playerOnlineID_5','teamID']).fillna(0).values

            records = team_stats.to_dict('records')

            # Iterate through the list in steps of two
            elo = EloRatingSystem()
            y_pred = []
            for i in range(0, len(records), 2):
                if i + 1 < len(records):
                    team100 = records[i]
                    team200 = records[i + 1]
                    days_since_match = team100['DaysSince']
                    tournament_value = team100['tournamentValue']
                    winner = team100['teamOnlineID'] if team100['winningTeam'] == 1 else team200['teamOnlineID']
                    loser = team100['teamOnlineID'] if team100['winningTeam'] == 0 else team200['teamOnlineID']
                    winner_data = team_stats_values[i] if team100['winningTeam'] == 1 else team_stats_values[i + 1]
                    loser_data = team_stats_values[i] if team100['winningTeam'] == 0 else team_stats_values[i + 1]
                    # winner_players_data = player_stats_values[i] if team100['winningTeam'] == 1 else player_stats_values[i + 1]
                    # loser_players_data = player_stats_values[i] if team100['winningTeam'] == 0 else player_stats_values[i + 1]
                    elo.add_player(team100['teamOnlineID'])
                    elo.add_player(team200['teamOnlineID'])
                    expected_score = elo.calculate_expected_score(team100['teamOnlineID'], team200['teamOnlineID'], False, False)
                    if expected_score != 0.5:
                        predicted = (team100['winningTeam'] == 1 and expected_score > 0.5) or (team100['winningTeam'] == 0 and expected_score < 0.5)
                        y_pred.append(predicted)
                    performance_factor_winner,performance_factor_loser = performance_calculator.calculate_team_performance(winner_data,loser_data)
                    # performance_winner_players,performance_loser_players = performance_calculator.calculate_players_performance(winner_players_data,loser_players_data)
                    elo.update_ratings(
                        winner, 
                        loser, 
                        performance_factor_winner, 
                        performance_factor_loser, 
                        tournament_value,
                        days_since_match,
                        False,
                        False
                    )
            ratings = elo.get_all_ratings()
            rank = 0
            last_score = -1
            last_unknown = 1
            ranking = []
            for team_id,score in ratings:
                if team_id not in team_ids:
                    continue
                if str(team_id) in teams_mapping:
                    team_esports_data = teams_mapping[str(team_id)] 
                else: 
                    team_esports_data = {
                        'name':'unknown_'+ str(last_unknown),
                        'acronym': 'UNK' + str(last_unknown)
                        }
                    last_unknown += 1
                if int(score) != int(last_score):
                    rank += 1
                ranking.append({
                    'team_id': team_id, 
                    'team_code': team_esports_data['acronym'], 
                    'team_name': team_esports_data['name'], 
                    'rank': rank, 
                })
            # Return the result as a JSON response
            response = {
                'statusCode': 200,
                'body': {'result': ranking}
            }
        except Exception as e:
            response = {
                'statusCode': 400,
                'body': {'error': str(e)}
            }
    else:
        # If 'a' or 'b' is missing, return an error response
        response = {
            'statusCode': 400,
            'body': {'error': 'Missing parameter "tournament_id"'}
        }
    return response