from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import sqlite3
from dataclasses import dataclass


class EventType(Enum):
    Possession = "possession"


def _possession_parse(xml_data):
    if xml_data is not None:
        try:
            root = ET.fromstring(xml_data)
            final_home_possession, final_away_possession = None, None
            for value in root.findall(".//value"):
                elapsed = value.find("elapsed")
                if elapsed is not None and elapsed.text == "90":
                    home_possession_tag = value.find("homepos")
                    away_possession_tag = value.find("awaypos")

                    if (
                        home_possession_tag is not None
                        and away_possession_tag is not None
                    ):
                        final_home_possession = int(home_possession_tag.text)
                        final_away_possession = int(away_possession_tag.text)

            return final_home_possession, final_away_possession
        except ET.ParseError:
            return None, None
    return None, None


def parse_goals(xml_data, match_api_id):
    goals_data = []

    if xml_data is not None:
        root = ET.fromstring(xml_data)
        for value in root.findall(".//value"):
            team = value.find("team")
            elapsed = value.find("elapsed")
            player1 = value.find("player1")
            player2 = value.find("player2")
            goal_type = value.find("goal_type")

            if (
                team is not None
                and elapsed is not None
                and player1 is not None
                and goal_type is not None
            ):
                if (
                    goal_type.text != "n"
                    and goal_type.text != "o"
                    and goal_type.text != "p"
                ):
                    # npm - penalties that were saved/missed
                    continue

                goal_data = {
                    "match_api_id": match_api_id,
                    "team_id": int(team.text),
                    "game_min": int(elapsed.text),
                    "scoring_player_id": int(player1.text),
                    "assist_player_id": int(player2.text)
                    if player2 is not None
                    else None,
                    "goal_type": goal_type.text,
                }
                goals_data.append(goal_data)

    return goals_data


def event_parser(_df):
    """
    Some stats like 'possession' are encoded using XML which contain multiple observations for the stat during the game.
    Instead, we'll split it into  "home" and "away" columns (like for goals) and use the value from last possession observation (min=90)
    :return:
    """
    # if ev_type == EventType.Possession:
    _df["home_possession"], _df["away_possession"] = zip(
        *_df["possession"].apply(_possession_parse)
    )
    _df = _df.drop(columns="possession")

    return _df


@dataclass
class LoadedDataData:
    matches_df: pd.DataFrame
    team_attrs_df: pd.DataFrame
    teams_df: pd.DataFrame
    league_df: pd.DataFrame
    country_df: pd.DataFrame
    team_attrs_features_df_encoded: pd.DataFrame
    player_df: Optional[pd.DataFrame]
    goals_df: Optional[pd.DataFrame]
    player_attr_df: Optional[pd.DataFrame]

    DEBUG_team_lague_names: Optional[pd.DataFrame] = None

    @property
    def matches_df_short(self):
        SUBSET = [
            "id",
            "days_after_first_date",
            "home_team_goal",
            "stage",
            "league_name",
            "season_start_year",
            "away_team_goal",
            "match_api_id",
            "home_team_api_id",
            "away_team_api_id",
            "home_win_odds",
            "draw_odds",
            "away_win_odds",
        ]
        return self.matches_df[SUBSET]


FEATURES_ENCODE = [
    "buildUpPlaySpeed",
    "buildUpPlaySpeedClass",
    "buildUpPlayDribbling",
    "buildUpPlayDribblingClass",
    "buildUpPlayPassing",
    "buildUpPlayPassingClass",
    "buildUpPlayPositioningClass",
    "chanceCreationPassing",
    "chanceCreationPassingClass",
    "chanceCreationCrossing",
    "chanceCreationCrossingClass",
    "chanceCreationShooting",
    "chanceCreationShootingClass",
    "chanceCreationPositioningClass",
    "defencePressure",
    "defencePressureClass",
    "defenceAggression",
    "defenceAggressionClass",
    "defenceTeamWidth",
    "defenceTeamWidthClass",
    "defenceDefenderLineClass",
]

FIRST_DATE = np.datetime64("2008-07-18T00:00:00.000000000")


# FIRST_DATE = np.datetime64('2008-07-18T00:00:00.000000000')


def _convert_dates(_df):
    _df["date"] = pd.to_datetime(_df["date"])

    first_date = FIRST_DATE
    # first_date = _df['date'].min()
    _df["days_after_first_date"] = (_df["date"] - first_date).dt.days

    if "season" in _df:
        _df["season_start_year"] = _df["season"].str.split("/").str[0]
        _df = _df.drop(["season"], axis=1)
    # _df = _df.drop(['date'], axis=1)

    return _df


BETTING_ODDS_VALUE = {
    "B365H": "Bet365 home win odds",
    "B365D": "Bet365 draw odds",
    "B365A": "Bet365 away win odds",
    "BSH": "Blue Square home win odds",
    "BSD": "Blue Square draw odds",
    "BSA": "Blue Square away win odds",
    "BWH": "Bet&Win home win odds",
    "BWD": "Bet&Win draw odds",
    "BWA": "Bet&Win away win odds",
    "GBH": "Gamebookers home win odds",
    "GBD": "Gamebookers draw odds",
    "GBA": "Gamebookers away win odds",
    "IWH": "Interwetten home win odds",
    "IWD": "Interwetten draw odds",
    "IWA": "Interwetten away win odds",
    "LBH": "Ladbrokes home win odds",
    "LBD": "Ladbrokes draw odds",
    "LBA": "Ladbrokes away win odds",
    "PSH": "Pinnacle home win odds",
    "PH": "Pinnacle home win odds",
    "PSD": "Pinnacle draw odds",
    "PD": "Pinnacle draw odds",
    "PSA": "Pinnacle away win odds",
    "PA": "Pinnacle away win odds",
    "SOH": "Sporting Odds home win odds",
    "SOD": "Sporting Odds draw odds",
    "SOA": "Sporting Odds away win odds",
    "SBH": "Sportingbet home win odds",
    "SBD": "Sportingbet draw odds",
    "SBA": "Sportingbet away win odds",
    "SJH": "Stan James home win odds",
    "SJD": "Stan James draw odds",
    "SJA": "Stan James away win odds",
    "SYH": "Stanleybet home win odds",
    "SYD": "Stanleybet draw odds",
    "SYA": "Stanleybet away win odds",
    "VCH": "VC Bet home win odds",
    "VCD": "VC Bet draw odds",
    "VCA": "VC Bet away win odds",
    "WHH": "William Hill home win odds",
    "WHD": "William Hill draw odds",
    "WHA": "William Hill away win odds",
    "MaxH": "Market maximum home win odds",
    "MaxD": "Market maximum draw win odds",
    "MaxA": "Market maximum away win odds",
    "AvgH": "Market average home win odds",
    "AvgD": "Market average draw win odds",
    "AvgA": "Market average away win odds",
}


def add_average_odds_columns(_df):
    home_win_cols = [
        col
        for col, desc in BETTING_ODDS_VALUE.items()
        if "home win odds" in desc and col in _df.columns
    ]
    draw_cols = [
        col
        for col, desc in BETTING_ODDS_VALUE.items()
        if "draw odds" in desc and col in _df.columns
    ]
    away_win_cols = [
        col
        for col, desc in BETTING_ODDS_VALUE.items()
        if "away win odds" in desc and col in _df.columns
    ]

    _df["home_win_odds"] = _df[home_win_cols].mean(axis=1, skipna=True)
    _df["draw_odds"] = _df[draw_cols].mean(axis=1, skipna=True)
    _df["away_win_odds"] = _df[away_win_cols].mean(axis=1, skipna=True)

    return _df


def load_data(inc_players=False):
    con = sqlite3.connect("../dataset/database.sqlite")
    matches_df = pd.read_sql("SELECT * from Match", con)
    team_attrs_df = pd.read_sql("SELECT * from Team_Attributes", con)
    teams_df = pd.read_sql("SELECT * from Team", con)
    league_df = pd.read_sql("SELECT * from League", con)
    country_df = pd.read_sql("SELECT * from Country", con)

    player_df = None
    player_attr_df = None
    if inc_players:
        player_df = pd.read_sql("SELECT * from Player", con)

        player_attr_df = pd.read_sql("SELECT * from Player_Attributes", con)
        # player_attr_df["player_api_id"] = player_attr_df["player_api_id"].astype('Int64') TODO: no need for nullable
        player_attr_df = _convert_dates(player_attr_df)

    # Drop the original 'date' and 'season' columns if no longer needed
    matches_df = _convert_dates(matches_df)
    team_attrs_df = _convert_dates(team_attrs_df)

    # league_df_sorted = league_df.sort_values('id', ascending=False)

    # Function to determine league based on team's id
    # def get_league_name(team_id):
    #     for _, row in league_df_sorted.iterrows():
    #         if team_id >= row['id']:
    #             return row['name']
    #     return 'Unknown'
    #
    # # Apply function to teams_df
    # teams_df['league_name'] = teams_df['id'].apply(get_league_name)
    # teams_df['league_name'].value_counts()

    matches_df["season_start_year"] = matches_df["season_start_year"].astype(int)

    team_attrs_df = pd.merge(
        team_attrs_df,
        teams_df[["team_api_id", "team_long_name"]],
        on="team_api_id",
        how="left",
        suffixes=("", ""),
    )

    team_league_table = (
        matches_df.groupby(["home_team_api_id", "league_id"])
        .count()["id"]
        .reset_index()
    )
    team_league_table = team_league_table.rename(
        columns={"home_team_api_id": "team_api_id"}
    )
    team_league_table = team_league_table.merge(
        league_df, how="left", left_on="league_id", right_on="id"
    )
    team_league_table = team_league_table.rename(columns={"name": "league_name"})

    team_league_names = league_df[["id", "name"]].rename(
        columns={"id": "league_id", "name": "league_name"}
    )
    matches_df = matches_df.merge(team_league_names, how="left", on="league_id")

    team_attrs_df = pd.merge(
        team_attrs_df,
        team_league_table,
        on="team_api_id",
        how="left",
        suffixes=("", ""),
    )

    team_attrs_df = team_attrs_df.sort_values(
        by=["team_api_id", "days_after_first_date"], ascending=True
    )
    features = [
        "days_after_first_date",
        "league_name",
        "team_long_name",
        "team_api_id",
        *FEATURES_ENCODE,
    ]

    # TODO: implement scaling in separate pipeline
    team_attrs_features_df = team_attrs_df[features]
    # team_attrs_features_df = team_attrs_df

    categorical_cols = (
        team_attrs_features_df[FEATURES_ENCODE]
        .select_dtypes(include=["object"])
        .columns
    )

    one_hot_encoded_data = pd.get_dummies(
        team_attrs_features_df[categorical_cols], drop_first=True
    )

    team_attrs_features_df_encoded = team_attrs_features_df.drop(
        categorical_cols, axis=1
    )
    team_attrs_features_df_encoded = pd.concat(
        [team_attrs_features_df_encoded, one_hot_encoded_data], axis=1
    )
    team_attrs_features_df_encoded.dropna(axis=1, inplace=True)

    # Parse additional match stats
    matches_df = event_parser(matches_df)

    # Cast player ids to int
    player_cols = [
        c
        for c in matches_df.columns
        if "_player_" in c and not ("_Y" in c or "_X" in c)
    ]
    for col in player_cols:
        matches_df[col] = matches_df[col].astype("Int64")

    matches_df["all_players_in_game"] = matches_df[player_cols].values.tolist()

    # GOALS::
    # Apply the function to each row and create a Series of lists
    goal_data_series = matches_df.apply(
        lambda row: parse_goals(row["goal"], row["match_api_id"]), axis=1
    )

    # Explode the Series and normalize the data into a DataFrame
    goals_df = pd.json_normalize(goal_data_series.explode()).dropna(
        subset=["match_api_id"]
    )
    goals_df["scoring_player_id"] = goals_df["scoring_player_id"].astype("Int64")
    goals_df["assist_player_id"] = goals_df["assist_player_id"].astype("Int64")
    goals_df["match_api_id"] = goals_df["match_api_id"].astype("Int64")

    # scoring_player_id
    # ': int(player1.text),
    # 'assist_player_id

    matches_df = add_average_odds_columns(matches_df)

    return LoadedDataData(
        matches_df=matches_df,
        team_attrs_df=team_attrs_df,
        teams_df=teams_df,
        league_df=league_df,
        country_df=country_df,
        team_attrs_features_df_encoded=team_attrs_features_df_encoded,
        goals_df=goals_df,
        player_df=player_df,
        player_attr_df=player_attr_df,
        DEBUG_team_lague_names=team_league_names,
    )
