import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def normalize_dataset(df,columns_to_drop):
    features = df.drop(columns=columns_to_drop)  # Remove non-numeric columns
    scaler_teams = StandardScaler()
    normalized = pd.DataFrame(scaler_teams.fit_transform(features), columns=features.columns)
    return normalized,scaler_teams

def normalize_processed_dataset(df):
    # features = df.drop(columns=columns_to_drop)  # Remove non-numeric columns
    scaler_teams = StandardScaler()
    normalized = pd.DataFrame(scaler_teams.fit_transform(df), columns=df.columns)
    return normalized,scaler_teams

def find_elbow_point(data, max_components=None):
    if max_components is None:
        max_components = min(data.shape)

    pca = PCA(n_components=max_components)
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_
    
    # Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    
    # Plot the cumulative explained variance
    # plt.plot(range(1, max_components + 1), explained_variance, marker='o', linestyle='--')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.title('PCA Elbow Method')
    
    # Find the elbow point (the point with diminishing returns)
    diff = np.diff(cumulative_variance)
    elbow_point = np.where(diff < 0.01)[0][0] + 1  # You can adjust the threshold (0.01)

    # plt.vlines(elbow_point, 0, 1, colors='r', linestyles='dashed', label=f'Elbow ({elbow_point} components)')
    # plt.legend()
    
    # plt.show()
    
    return elbow_point

teams_df = pd.read_csv('data/games_teams_extracted_features.csv',sep=';')
teams_exploded_df = pd.read_csv('tournament_games_team_stats_vff.csv',sep=';')
players_df = pd.read_csv('data/games_players_extracted_features.csv',sep=';')
teams_id_columns = ['platformGameID', 'teamOnlineID_100', 'teamOnlineID_200', 'winningTeam']
players_id_columns = ['teamID_1','teamID_2','teamID_3','teamID_4','teamID_5','teamID_6','teamID_7','teamID_8','teamID_9','teamID_10','Champion_1','Champion_2','Champion_3','Champion_4','Champion_5','Champion_6','Champion_7','Champion_8','Champion_9','Champion_10','platformGameID', 'teamOnlineID_100', 'teamOnlineID_200','playerOnlineID_1','playerOnlineID_2','playerOnlineID_3','playerOnlineID_4','playerOnlineID_5','playerOnlineID_6','playerOnlineID_7','playerOnlineID_8','playerOnlineID_9','playerOnlineID_10','winningTeam']
players_useless_columns = ['goldFromkillWard_1','goldFromkillWard_2','goldFromkillWard_3','goldFromkillWard_4','goldFromkillWard_5','goldFromkillWard_6','goldFromkillWard_7','goldFromkillWard_8','goldFromkillWard_9','goldFromkillWard_10','BASIC_PINGS_1','BASIC_PINGS_2','BASIC_PINGS_3','BASIC_PINGS_4','BASIC_PINGS_5','BASIC_PINGS_6','BASIC_PINGS_7','BASIC_PINGS_8','BASIC_PINGS_9','BASIC_PINGS_10','COMMAND_PINGS_1','COMMAND_PINGS_2','COMMAND_PINGS_3','COMMAND_PINGS_4','COMMAND_PINGS_5','COMMAND_PINGS_6','COMMAND_PINGS_7','COMMAND_PINGS_8','COMMAND_PINGS_9','COMMAND_PINGS_10','DANGER_PINGS_1','DANGER_PINGS_2','DANGER_PINGS_3','DANGER_PINGS_4','DANGER_PINGS_5','DANGER_PINGS_6','DANGER_PINGS_7','DANGER_PINGS_8','DANGER_PINGS_9','DANGER_PINGS_10','GET_BACK_PINGS_1','GET_BACK_PINGS_2','GET_BACK_PINGS_3','GET_BACK_PINGS_4','GET_BACK_PINGS_5','GET_BACK_PINGS_6','GET_BACK_PINGS_7','GET_BACK_PINGS_8','GET_BACK_PINGS_9','GET_BACK_PINGS_10','RETREAT_PINGS_1','RETREAT_PINGS_2','RETREAT_PINGS_3','RETREAT_PINGS_4','RETREAT_PINGS_5','RETREAT_PINGS_6','RETREAT_PINGS_7','RETREAT_PINGS_8','RETREAT_PINGS_9','RETREAT_PINGS_10','ON_MY_WAY_PINGS_1','ON_MY_WAY_PINGS_2','ON_MY_WAY_PINGS_3','ON_MY_WAY_PINGS_4','ON_MY_WAY_PINGS_5','ON_MY_WAY_PINGS_6','ON_MY_WAY_PINGS_7','ON_MY_WAY_PINGS_8','ON_MY_WAY_PINGS_9','ON_MY_WAY_PINGS_10','ASSIST_ME_PINGS_1','ASSIST_ME_PINGS_2','ASSIST_ME_PINGS_3','ASSIST_ME_PINGS_4','ASSIST_ME_PINGS_5','ASSIST_ME_PINGS_6','ASSIST_ME_PINGS_7','ASSIST_ME_PINGS_8','ASSIST_ME_PINGS_9','ASSIST_ME_PINGS_10','ENEMY_MISSING_PINGS_1','ENEMY_MISSING_PINGS_2','ENEMY_MISSING_PINGS_3','ENEMY_MISSING_PINGS_4','ENEMY_MISSING_PINGS_5','ENEMY_MISSING_PINGS_6','ENEMY_MISSING_PINGS_7','ENEMY_MISSING_PINGS_8','ENEMY_MISSING_PINGS_9','ENEMY_MISSING_PINGS_10','PUSH_PINGS_1','PUSH_PINGS_2','PUSH_PINGS_3','PUSH_PINGS_4','PUSH_PINGS_5','PUSH_PINGS_6','PUSH_PINGS_7','PUSH_PINGS_8','PUSH_PINGS_9','PUSH_PINGS_10','ALL_IN_PINGS_1','ALL_IN_PINGS_2','ALL_IN_PINGS_3','ALL_IN_PINGS_4','ALL_IN_PINGS_5','ALL_IN_PINGS_6','ALL_IN_PINGS_7','ALL_IN_PINGS_8','ALL_IN_PINGS_9','ALL_IN_PINGS_10','HOLD_PINGS_1','HOLD_PINGS_2','HOLD_PINGS_3','HOLD_PINGS_4','HOLD_PINGS_5','HOLD_PINGS_6','HOLD_PINGS_7','HOLD_PINGS_8','HOLD_PINGS_9','HOLD_PINGS_10','BAIT_PINGS_1','BAIT_PINGS_2','BAIT_PINGS_3','BAIT_PINGS_4','BAIT_PINGS_5','BAIT_PINGS_6','BAIT_PINGS_7','BAIT_PINGS_8','BAIT_PINGS_9','BAIT_PINGS_10','VISION_CLEARED_PINGS_1','VISION_CLEARED_PINGS_2','VISION_CLEARED_PINGS_3','VISION_CLEARED_PINGS_4','VISION_CLEARED_PINGS_5','VISION_CLEARED_PINGS_6','VISION_CLEARED_PINGS_7','VISION_CLEARED_PINGS_8','VISION_CLEARED_PINGS_9','VISION_CLEARED_PINGS_10','ENEMY_VISION_PINGS_1','ENEMY_VISION_PINGS_2','ENEMY_VISION_PINGS_3','ENEMY_VISION_PINGS_4','ENEMY_VISION_PINGS_5','ENEMY_VISION_PINGS_6','ENEMY_VISION_PINGS_7','ENEMY_VISION_PINGS_8','ENEMY_VISION_PINGS_9','ENEMY_VISION_PINGS_10','NEED_VISION_PINGS_1','NEED_VISION_PINGS_2','NEED_VISION_PINGS_3','NEED_VISION_PINGS_4','NEED_VISION_PINGS_5','NEED_VISION_PINGS_6','NEED_VISION_PINGS_7','NEED_VISION_PINGS_8','NEED_VISION_PINGS_9','NEED_VISION_PINGS_10','PERK0_1','PERK0_2','PERK0_3','PERK0_4','PERK0_5','PERK0_6','PERK0_7','PERK0_8','PERK0_9','PERK0_10','PERK0_VAR1_1','PERK0_VAR1_2','PERK0_VAR1_3','PERK0_VAR1_4','PERK0_VAR1_5','PERK0_VAR1_6','PERK0_VAR1_7','PERK0_VAR1_8','PERK0_VAR1_9','PERK0_VAR1_10','PERK0_VAR2_1','PERK0_VAR2_2','PERK0_VAR2_3','PERK0_VAR2_4','PERK0_VAR2_5','PERK0_VAR2_6','PERK0_VAR2_7','PERK0_VAR2_8','PERK0_VAR2_9','PERK0_VAR2_10','PERK0_VAR3_1','PERK0_VAR3_2','PERK0_VAR3_3','PERK0_VAR3_4','PERK0_VAR3_5','PERK0_VAR3_6','PERK0_VAR3_7','PERK0_VAR3_8','PERK0_VAR3_9','PERK0_VAR3_10','PERK1_1','PERK1_2','PERK1_3','PERK1_4','PERK1_5','PERK1_6','PERK1_7','PERK1_8','PERK1_9','PERK1_10','PERK1_VAR1_1','PERK1_VAR1_2','PERK1_VAR1_3','PERK1_VAR1_4','PERK1_VAR1_5','PERK1_VAR1_6','PERK1_VAR1_7','PERK1_VAR1_8','PERK1_VAR1_9','PERK1_VAR1_10','PERK1_VAR2_1','PERK1_VAR2_2','PERK1_VAR2_3','PERK1_VAR2_4','PERK1_VAR2_5','PERK1_VAR2_6','PERK1_VAR2_7','PERK1_VAR2_8','PERK1_VAR2_9','PERK1_VAR2_10','PERK1_VAR3_1','PERK1_VAR3_2','PERK1_VAR3_3','PERK1_VAR3_4','PERK1_VAR3_5','PERK1_VAR3_6','PERK1_VAR3_7','PERK1_VAR3_8','PERK1_VAR3_9','PERK1_VAR3_10','PERK2_1','PERK2_2','PERK2_3','PERK2_4','PERK2_5','PERK2_6','PERK2_7','PERK2_8','PERK2_9','PERK2_10','PERK2_VAR1_1','PERK2_VAR1_2','PERK2_VAR1_3','PERK2_VAR1_4','PERK2_VAR1_5','PERK2_VAR1_6','PERK2_VAR1_7','PERK2_VAR1_8','PERK2_VAR1_9','PERK2_VAR1_10','PERK2_VAR2_1','PERK2_VAR2_2','PERK2_VAR2_3','PERK2_VAR2_4','PERK2_VAR2_5','PERK2_VAR2_6','PERK2_VAR2_7','PERK2_VAR2_8','PERK2_VAR2_9','PERK2_VAR2_10','PERK2_VAR3_1','PERK2_VAR3_2','PERK2_VAR3_3','PERK2_VAR3_4','PERK2_VAR3_5','PERK2_VAR3_6','PERK2_VAR3_7','PERK2_VAR3_8','PERK2_VAR3_9','PERK2_VAR3_10','PERK3_1','PERK3_2','PERK3_3','PERK3_4','PERK3_5','PERK3_6','PERK3_7','PERK3_8','PERK3_9','PERK3_10','PERK3_VAR1_1','PERK3_VAR1_2','PERK3_VAR1_3','PERK3_VAR1_4','PERK3_VAR1_5','PERK3_VAR1_6','PERK3_VAR1_7','PERK3_VAR1_8','PERK3_VAR1_9','PERK3_VAR1_10','PERK3_VAR2_1','PERK3_VAR2_2','PERK3_VAR2_3','PERK3_VAR2_4','PERK3_VAR2_5','PERK3_VAR2_6','PERK3_VAR2_7','PERK3_VAR2_8','PERK3_VAR2_9','PERK3_VAR2_10','PERK3_VAR3_1','PERK3_VAR3_2','PERK3_VAR3_3','PERK3_VAR3_4','PERK3_VAR3_5','PERK3_VAR3_6','PERK3_VAR3_7','PERK3_VAR3_8','PERK3_VAR3_9','PERK3_VAR3_10','PERK4_1','PERK4_2','PERK4_3','PERK4_4','PERK4_5','PERK4_6','PERK4_7','PERK4_8','PERK4_9','PERK4_10','PERK4_VAR1_1','PERK4_VAR1_2','PERK4_VAR1_3','PERK4_VAR1_4','PERK4_VAR1_5','PERK4_VAR1_6','PERK4_VAR1_7','PERK4_VAR1_8','PERK4_VAR1_9','PERK4_VAR1_10','PERK4_VAR2_1','PERK4_VAR2_2','PERK4_VAR2_3','PERK4_VAR2_4','PERK4_VAR2_5','PERK4_VAR2_6','PERK4_VAR2_7','PERK4_VAR2_8','PERK4_VAR2_9','PERK4_VAR2_10','PERK4_VAR3_1','PERK4_VAR3_2','PERK4_VAR3_3','PERK4_VAR3_4','PERK4_VAR3_5','PERK4_VAR3_6','PERK4_VAR3_7','PERK4_VAR3_8','PERK4_VAR3_9','PERK4_VAR3_10','PERK5_1','PERK5_2','PERK5_3','PERK5_4','PERK5_5','PERK5_6','PERK5_7','PERK5_8','PERK5_9','PERK5_10','PERK5_VAR1_1','PERK5_VAR1_2','PERK5_VAR1_3','PERK5_VAR1_4','PERK5_VAR1_5','PERK5_VAR1_6','PERK5_VAR1_7','PERK5_VAR1_8','PERK5_VAR1_9','PERK5_VAR1_10','PERK5_VAR2_1','PERK5_VAR2_2','PERK5_VAR2_3','PERK5_VAR2_4','PERK5_VAR2_5','PERK5_VAR2_6','PERK5_VAR2_7','PERK5_VAR2_8','PERK5_VAR2_9','PERK5_VAR2_10','PERK5_VAR3_1','PERK5_VAR3_2','PERK5_VAR3_3','PERK5_VAR3_4','PERK5_VAR3_5','PERK5_VAR3_6','PERK5_VAR3_7','PERK5_VAR3_8','PERK5_VAR3_9','PERK5_VAR3_10']

teams_exploded_data = teams_exploded_df.drop(columns=[
            'platformGameID','teamOnlineID', 'tournamentID', 'tournamentName',
            'tournamentSlug', 'tournamentCategory', 'tournamentStartDate',
            'tournamentEndDate', 'stageName', 'sectionName', 'matchID','winningTeam',
            'esportsGameID', 'teamID', 'matchTime','teamID'])

teams_features, teams_scaler = normalize_dataset(teams_df,teams_id_columns)
players_features = normalize_dataset(players_df,players_id_columns+players_useless_columns)
players_features.fillna(0,inplace=True)

elbow_point = find_elbow_point(players_features,max_components=50)

# pca_teams = PCA(n_components=2)
# principalComponents_teams = pca_teams.fit_transform(teams_df.values)

pca_players = PCA(n_components=elbow_point)
principalComponents_players = pca_players.fit_transform(players_features)

pca_players_df = pd.DataFrame(data=principalComponents_players, columns=[f'PC{i+1}' for i in range(elbow_point)])
for column in players_id_columns:
    pca_players_df[column] = players_df[column]
    players_features[column] = players_df[column]

for column in teams_id_columns:
    teams_features[column] = teams_df[column]
    

winning_team_dict = {
    '100':0,
    '200':1
}

pca_players_df['winningTeam'] = pca_players_df['winningTeam'].apply(lambda x: winning_team_dict[str(x)])
teams_features['winningTeam'] = teams_features['winningTeam'].apply(lambda x: winning_team_dict[str(x)])
players_features['winningTeam'] = players_features['winningTeam'].apply(lambda x: winning_team_dict[str(x)])

pca_players_df.to_csv('players_stats_pca.csv',sep=';',index=False)
teams_features.to_csv('norm_teams_stats.csv',sep=';',index=False)
players_features.to_csv('norm_players_stats.csv',sep=';',index=False)
