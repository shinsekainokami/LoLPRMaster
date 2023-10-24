import pandas as pd
import json

with open('data/esports-data/tournaments.json') as f:
    tournaments_data = json.load(f)

with open('data/esports-data/mapping_data.json') as f:
    mappings_data = json.load(f)

mappings = {
    esports_game["esportsGameId"]: esports_game for esports_game in mappings_data
}

tournament_classification = {'worlds':'major',
'msi':'major',
'lck':'regional',
'lec':'regional',
'lcs':'regional',
'lpl':'regional',
'ljl':'regional',
'pcs':'regional',
'cblol_academy':'academy',
'lcs_academy':'academy',
'lla_promotion':'academy',
'lcs_amateur_circuit':'academy',
'ljl_academy':'academy',
'eumasters':'academy',
'ultraliga':'academy',
'lcs_lock_in':'offseason',
'lck_regional_finals':'offseason',
'lla_closing':'offseason',
'eumasters_play_ins':'offseason',
'eumasters_super_finals':'offseason',
'lla_promotion':'offseason',
'eumasters':'offseason',
'ultraliga':'offseason',
'gll':'other',
'elite_series':'other',
'superliga':'other',
'tcl':'other',
'nlc':'other',
'pg_nationals':'other',
'ebl':'other',
'lrs_closing':'other',
'liga_portuguesa':'other',
'elements_league':'other',
'hitpoint_masters':'other',
'tal':'other',
'belgian_league':'other',
'dutch_league':'other',
'opl':'other',
'golden_league':'other',
'esports_balkan_league':'other',
'lrn_closing':'other',
'lco_split':'other',
'nacl':'other',
'vcs':'other',
'ddh':'other',
'lck_challengers':'other',
'hitpoint_masters':'other',
'stars_league':'other',
'master_flow_league':'other',
'vdl':'other',
'lck_challengers':'other',
'eu_masters_play_ins':'other',
'eumasters_super_finals':'other',
'lla':'other',
'honor_division':'other',
'lec_season_finals':'other',
'lfl':'other',
'gll':'other'}

def get_category_from_slug(slug):
    for tournament,category in tournament_classification.items():
        if tournament in slug:
            return category
    return 'unknown'

game_context_data = []
for tournament in tournaments_data:
    tournamentID = tournament['id']
    tournamentStartDate = tournament['startDate']
    tournamentEndDate = tournament['endDate']
    tournamentName = tournament['name']
    tournamentSlug = tournament['slug']
    tournamentCategory = get_category_from_slug(tournamentSlug)
    for stage in tournament["stages"]:
        stageName = stage['slug']
        for section in stage["sections"]:
            sectionName = section["name"]
            for match in section["matches"]:
                matchID = match['id']
                for game in match["games"]:
                    if game['state'] == 'completed':
                        esportsGameId = game['id']
                        platformGameID = mappings[game["id"]]["platformGameId"] if game["id"] in mappings else None
                        game_context_row = {
                            'tournamentID': tournamentID,
                            'tournamentName': tournamentName,
                            'tournamentSlug': tournamentSlug,
                            'tournamentCategory': tournamentCategory,
                            'tournamentStartDate': tournamentStartDate,
                            'tournamentEndDate': tournamentEndDate,
                            'stageName': stageName,
                            'sectionName': sectionName,
                            'matchID': matchID,
                            'esportsGameID': esportsGameId,
                            'platformGameID': platformGameID
                        }
                        game_context_data.append(game_context_row)
                    else:
                        continue

game_context_df = pd.DataFrame(game_context_data)
game_context_df.to_csv('game_context_data.csv', sep=';', index=False)

games_team_stats = pd.read_csv('./data/games_teams_extracted_features.csv',sep=';')
games_player_stats = pd.read_csv('./data/games_players_extracted_features.csv',sep=';')

match_datetime = pd.read_csv('./data/teams_extracted_features.csv',sep=';')

tournament_games_team_stats = games_team_stats.merge(game_context_df,on='platformGameID', how='left')
tournament_games_player_stats = games_player_stats.merge(game_context_df,on='platformGameID', how='left')

regex_100 = r'.*(_100)'
regex_200 = r'.*(_200)'

tournament_games_team_100_stats = tournament_games_team_stats.filter(regex=f'^(?!{regex_200}).*$', axis=1)
tournament_games_team_100_stats.columns = [col.replace('_100','') for col in tournament_games_team_100_stats.columns]
tournament_games_team_100_stats['winningTeam'] = tournament_games_team_100_stats['winningTeam'].apply(lambda x: 1 if str(x)=='100' else 0)
tournament_games_team_100_stats['teamID'] = '100'

tournament_games_team_200_stats = tournament_games_team_stats.filter(regex=f'^(?!{regex_100}).*$', axis=1)
tournament_games_team_200_stats.columns = [col.replace('_200','') for col in tournament_games_team_200_stats.columns]
tournament_games_team_200_stats['winningTeam'] = tournament_games_team_200_stats['winningTeam'].apply(lambda x: 1 if str(x)=='200' else 0)
tournament_games_team_200_stats['teamID'] = '200'

tournament_games_exploded_teams_stats = pd.concat([tournament_games_team_100_stats,tournament_games_team_200_stats]).sort_index().reset_index(drop=True)

regex_pl_100 = r'.*(_100|_[1-5](?![0-9]))'
regex_pl_200 = r'.*(_200|_6|_7|_8|_9|_10(?![0-9]))'

tournament_games_player_100_stats = tournament_games_player_stats.filter(regex=f'^(?!{regex_pl_200}).*$', axis=1)
tournament_games_player_100_stats.columns = [col.replace('_100', '') for col in tournament_games_player_100_stats.columns]
tournament_games_player_100_stats['winningTeam'] = tournament_games_player_100_stats['winningTeam'].apply(lambda x: 1 if str(x)=='100' else 0)

tournament_games_player_200_stats = tournament_games_player_stats.filter(regex=f'^(?!{regex_pl_100}).*$', axis=1)
tournament_games_player_200_stats.columns = [col.replace('_200', '').replace('_10', '_5').replace('_9', '_4').replace('_8', '_3').replace('_7', '_2').replace('_6', '_1') for col in tournament_games_player_200_stats.columns]
tournament_games_player_200_stats['winningTeam'] = tournament_games_player_200_stats['winningTeam'].apply(lambda x: 1 if str(x)=='200' else 0)

tournament_games_exploded_players_stats = pd.concat([tournament_games_player_100_stats,tournament_games_player_200_stats]).sort_index().reset_index(drop=True)
tournament_games_exploded_players_stats.drop(columns=['teamID_2','teamID_3','teamID_4','teamID_5'],inplace=True)
tournament_games_exploded_players_stats.rename({'teamID_1':'teamID'},axis=1, inplace=True)

tournaments_games_teams_data = tournament_games_exploded_teams_stats.merge(match_datetime[['platformGameID','matchTime']],on='platformGameID',how='left')
tournaments_games_players_data = tournament_games_exploded_players_stats.merge(match_datetime[['platformGameID','matchTime']],on='platformGameID',how='left')

tournament_games_exploded_teams_stats.to_csv('tournament_games_team_stats_vf.csv',sep=';',index=False)
tournament_games_exploded_players_stats.to_csv('tournament_games_player_stats_vf.csv',sep=';',index=False)

tournaments_games_teams_data.to_csv('tournament_games_team_stats_vff.csv',sep=';',index=False)
tournaments_games_players_data.to_csv('tournament_games_player_stats_vff.csv',sep=';',index=False)
