import time, os, json
import pandas as pd

def flatten_dataframe(df,column):
    melted_df = pd.melt(df, id_vars=[column], var_name='Stat', value_name='Value')
    melted_df['Stat'] = melted_df['Stat'] + '_' + melted_df[column].astype(str)
    # melted_df.set_index('Stat')['Value']
    headers = melted_df['Stat'].values
    values = melted_df['Value'].values
    flat_data_row = {}
    for header,value in zip(headers,values):
        flat_data_row[header] = value
    return flat_data_row

def get_tournament_id(gameName, team_100_id, team_200_id, tournaments_data):
    game_id = gameName.split('|')[0]
    for tournament in tournaments_data:
        for stage in tournament["stages"]:
            for section in stage["sections"]:
                for match in section["matches"]:
                    game_data = next((game for game in match["games"] if game['id'] == game_id), None)
                    if game_data != None:
                        return game_data['id']
                        # team_100_players = next((team for team in match["teams"] if team['id'] == team_100_id), None)
                        # team_200_players = next((team for team in match["teams"] if team['id'] == team_200_id), None)
                        # tournament_game_data = {
                        #     'id':game_data['id'],
                        #     'team100_players':team_100_players,
                        #     'team200_players':team_200_players
                        # }
                        # return tournament_game_data
    return ''

def get_game_data(game,map_data,tournaments_data):
    # print(f'Reading file {game}...')
    start_time = time.time()
    with open(game, 'r') as f:
        game_data = json.load(f)
    end_time = time.time()
    # print(f"Game file {game} read, {end_time-start_time}s...")
    game_meta = game_data[0]
    game_end = [event for event in game_data if event['eventType'] == 'game_end'][0]
    game_map = next((map for map in map_data if map['platformGameId'] == game_meta['platformGameId']),None)
    team_mapping = game_map['teamMapping']
    participant_mapping = game_map['participantMapping']
    last_stats_update = next((event for event in reversed(game_data) if event['eventType'] == 'stats_update'), None)
    team_stats_df = extract_team_features(last_stats_update)
    players_stats_df = extract_players_features(last_stats_update)
    team_stats_1d = flatten_dataframe(team_stats_df,'teamID')
    players_stats_1d = flatten_dataframe(players_stats_df,'participantID')
    team_stats_1d['platformGameID'] = game_end['platformGameId']
    team_stats_1d['gameID'] = game_end['gameName'].split('|')[0]
    team_stats_1d['teamOnlineID_100'] = team_mapping['100']
    team_stats_1d['teamOnlineID_200'] = team_mapping['200']
    team_stats_1d['winningTeam'] = game_end['winningTeam']
    team_stats_1d['matchTime'] = game_end['eventTime']
    team_stats_1d['tournamentID'] = get_tournament_id(game_end['gameName'],team_mapping['100'],team_mapping['200'],tournaments_data)
    players_stats_1d['platformGameID'] = game_end['platformGameId']
    players_stats_1d['gameID'] = game_end['gameName'].split('|')[0]
    players_stats_1d['teamOnlineID_100'] = team_mapping['100']
    players_stats_1d['teamOnlineID_200'] = team_mapping['200']
    for id,participant in sorted(participant_mapping.items(),key=lambda item: int(item[0])):
        players_stats_1d['playerOnlineID_'+str(id)] = participant
    players_stats_1d['winningTeam'] = game_end['winningTeam']
    players_stats_1d['matchTime'] = game_end['eventTime']
    players_stats_1d['tournamentID'] = get_tournament_id(game_end['gameName'],team_mapping['100'],team_mapping['200'],tournaments_data)
    return team_stats_1d, players_stats_1d

def extract_team_features(final_stats):
    teams_df = pd.DataFrame(final_stats['teams']).sort_values(by='teamID',ascending=True)
    return teams_df

def extract_players_features(final_stats):
    participants_data = final_stats['participants']
    participants_stats = []
    for participant in participants_data:
        participant_row = {}
        stats = participant['stats']
        participant_row['participantID'] = participant['participantID']
        participant_row['teamID'] = participant['teamID']
        for stat in stats:
            participant_row[stat['name']] = stat['value']
        gold_stats = participant['goldStats']
        for name, value in gold_stats.items():
            participant_row['goldFrom'+name] = value
        participant_row['level'] = participant['level']
        participant_row['healthMax'] = participant['healthMax']
        participant_row['XP'] = participant['XP']
        participant_row['Champion'] = participant['championName']
        participant_row['Gold'] = participant['totalGold']
        participant_row['abilityPower'] = participant['abilityPower']
        participant_row['armor'] = participant['armor']
        participant_row['attackSpeed'] = participant['attackSpeed']
        participant_row['magicResist'] = participant['magicResist']
        participants_stats.append(participant_row)
    players_df = pd.DataFrame(participants_stats).sort_values(by='participantID',ascending=True)
    return players_df

def construct_dataframes(gamefiles,map_data,tournaments_data):
    games_teams_data = []
    games_players_data = []
    i = 0
    e = 0
    gamefiles_not_processed = []
    start_time = time.time()
    for game in gamefiles:
        if i % 500 == 0:
            current_time = time.time() - start_time
            print(f"{i} game files processed. t:{current_time:.2f}s")
        try:
            team_stats_df, players_stats_df = get_game_data(game,map_data,tournaments_data)
            games_teams_data.append(team_stats_df)
            games_players_data.append(players_stats_df)
            print(f'Game file {game} processed.')
            i += 1
        except Exception as er:
            print(f"Game file {game} couldn't be processed. Exception: {er}")
            e += 1
            gamefiles_not_processed.append(game)
    end_time = time.time() - start_time
    print(f"All {i} game files processed ({e} with error). t:{end_time:.2f}s")
    games_teams_df = pd.DataFrame(games_teams_data)
    games_players_df = pd.DataFrame(games_players_data)
    return games_teams_df,games_players_df,gamefiles_not_processed

def main():
    print("Reading mapping data...")
    start_time = time.time()
    with open("./data/esports-data/mapping_data.json", 'r') as f:
        map_data = json.load(f)
    end_time = time.time()
    print(f"Mapping data read, {end_time-start_time}s...")
    print("Reading tournament data...")
    start_time = time.time()
    with open("./data/esports-data/tournaments.json", 'r') as f:
        tournaments_data = json.load(f)
    end_time = time.time()
    print(f"Tournament data read, {end_time-start_time}s...")
    gamefiles = ['./data/games/'+path for path in os.listdir('./data/games/')]
    games_teams_df,games_players_df,gamefiles_not_processed = construct_dataframes(gamefiles,map_data,tournaments_data)
    return games_teams_df,games_players_df,gamefiles_not_processed

games_teams_df,games_players_df,gamefiles_not_processed = main()
games_teams_df.to_csv('teams_extracted_features.csv', sep=';',header=True,index=False)
games_players_df.to_csv('players_extracted_features.csv', sep=';',header=True,index=False)

