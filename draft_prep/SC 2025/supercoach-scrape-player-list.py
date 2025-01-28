# Databricks notebook source
import pandas as pd
def flatten_position(column, requested_field):
    pos_list = []
    for pos in column:
        pos_list.append(pos[requested_field])
    return ' '.join(pos_list)


df_sc = pd.read_json('SC_2025_player_data_team_picker.json')
df_sc['position'] = df_sc['positions'].apply(lambda x: flatten_position(x, 'position'))
df_sc['long_team'] = df_sc['team'].apply(lambda x: x['name'])
df_sc['team'] = df_sc['team'].apply(lambda x: x['abbrev'])
# df_sc['adp'] = df_sc['player_stats'].apply(lambda x: x[0]['adp'])
df_sc

# COMMAND ----------

df_sc.to_csv('2025_SC_Player_list.csv')