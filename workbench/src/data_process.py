from dataclasses import dataclass

import pandas as pd

from workbench.src.data_loader import LoadedDataData


def __append_season_performance(dual_df):
    dual_df["points"] = dual_df["result"].map({1: 3, 0: 1, -1: 0})
    dual_df = dual_df.sort_values(by=["team_id", "season_start_year", "stage"])

    dual_df["cumulative_points"] = dual_df.groupby(["team_id", "season_start_year"])[
        "points"
    ].cumsum()

    dual_df["total_season_points_before"] = (
            dual_df["cumulative_points"] - dual_df["points"]
    )

    seasonal_points = (
        dual_df.groupby(["team_id", "season_start_year"])["points"].sum().reset_index()
    )

    seasonal_points.rename(
        columns={
            "points": "last_season_points",
            "season_start_year": "previous_season",
        },
        inplace=True,
    )
    seasonal_points["previous_season"] += 1

    dual_df = dual_df.merge(
        seasonal_points,
        how="left",
        left_on=["team_id", "season_start_year"],
        right_on=["team_id", "previous_season"],
    )

    dual_df.drop("previous_season", axis=1, inplace=True)
    # dual_df['last_season_points'].fillna(0, inplace=True)

    opponent_data = dual_df[
        [
            "team_id",
            "season_start_year",
            "stage",
            "total_season_points_before",
            "last_season_points",
        ]
    ].copy()
    opponent_data.rename(
        columns={
            "team_id": "opponent_id",
            "total_season_points_before": "opponent_total_season_points_before",
            "last_season_points": "opponent_last_season_points",
        },
        inplace=True,
    )

    dual_df = dual_df.merge(
        opponent_data,
        how="left",
        left_on=["opponent_id", "season_start_year", "stage"],
        right_on=["opponent_id", "season_start_year", "stage"],
    )
    return dual_df


def __apend_player_stats(data):
    player_attr_df = data.player_attr_df.sort_values(
        by=["player_api_id", "days_after_first_date"]
    )

    player_matches_melted = pd.melt(
        data.matches_df,
        id_vars=[
            "match_api_id",
            "days_after_first_date",
            "league_name",
            "season_start_year",
            "home_team_api_id",
            "away_team_api_id",
        ],
        value_vars=[f"home_player_{i}" for i in range(1, 12)]
                   + [f"away_player_{i}" for i in range(1, 12)],
        var_name="home_away",
        value_name="player_api_id",
    )

    player_matches_melted.dropna(
        subset=["player_api_id", "days_after_first_date"], inplace=True
    )

    player_matches_melted["player_api_id"] = player_matches_melted[
        "player_api_id"
    ].astype("int64")

    # player_matches_melted["player_api_id"].dtypes
    # player_attr_df["player_api_id"] = player_attr_df["player_api_id"].astype('Int64')
    #
    merged_df = pd.merge_asof(
        player_matches_melted.sort_values("days_after_first_date"),
        player_attr_df.sort_values("days_after_first_date"),
        by="player_api_id",
        on="days_after_first_date",
        direction="backward",
    )
    final_df = pd.merge(
        merged_df,
        data.player_df[["player_api_id", "player_name"]],
        on="player_api_id",
        how="left",
    )

    home_players = merged_df[merged_df["home_away"].str.contains("home")].copy()
    home_players["team_api_id"] = home_players["home_team_api_id"]
    home_players.drop("home_team_api_id", axis=1, inplace=True)
    home_players.drop("away_team_api_id", axis=1, inplace=True)

    away_players = merged_df[merged_df["home_away"].str.contains("away")].copy()
    away_players["team_api_id"] = away_players["away_team_api_id"]
    away_players.drop("away_team_api_id", axis=1, inplace=True)
    away_players.drop("home_team_api_id", axis=1, inplace=True)

    home_rating_sum = (
        home_players.groupby("match_api_id")["overall_rating"].sum().reset_index()
    )
    away_rating_sum = (
        away_players.groupby("match_api_id")["overall_rating"].sum().reset_index()
    )

    home_rating_sum.rename(columns={"overall_rating": "home_team_rating"}, inplace=True)
    away_rating_sum.rename(columns={"overall_rating": "away_team_rating"}, inplace=True)

    aggregated_df = pd.merge(home_rating_sum, away_rating_sum, on="match_api_id")
    player_matches_df = pd.concat([home_players, away_players], sort=False)

    player_matches_df
    return aggregated_df, player_matches_df


@dataclass
class RollingMatchTeamStats:
    dual_df: pd.DataFrame
    player_matches_df: pd.DataFrame


def append_rolling_match_team_stats(data: LoadedDataData):
    # matches_df_short = matches_df_short.copy()

    #
    aggregated_player_score_df, player_matches_df = __apend_player_stats(data)

    matches_df_short = data.matches_df_short.merge(
        aggregated_player_score_df, how="left", on="match_api_id"
    )
    # matches_df_short = data.matches_df_short
    home_df = matches_df_short.copy()

    home_df["result"] = pd.cut(
        home_df["home_team_goal"] - home_df["away_team_goal"],
        bins=[float("-inf"), -1, 0, float("inf")],
        labels=["loss", "draw", "win"],
    )

    home_df["goals_scored"] = home_df["home_team_goal"]
    home_df["goals_conceded"] = home_df["away_team_goal"]
    home_df["team_id"] = home_df["home_team_api_id"]
    home_df["opponent_id"] = home_df["away_team_api_id"]
    home_df[
        "is_home_team"
    ] = 1
    home_df["win_odds"] = home_df["home_win_odds"]
    home_df["opponent_win_odds"] = home_df["away_win_odds"]

    home_df["team_rating"] = home_df["home_team_rating"]
    home_df["opponent_team_rating"] = home_df["away_team_rating"]

    away_df = matches_df_short.copy()
    away_df["result"] = pd.cut(
        away_df["away_team_goal"] - away_df["home_team_goal"],
        bins=[float("-inf"), -1, 0, float("inf")],
        labels=["loss", "draw", "win"],
    )
    away_df["goals_scored"] = away_df["away_team_goal"]
    away_df["goals_conceded"] = away_df["home_team_goal"]
    away_df["team_id"] = away_df["away_team_api_id"]
    away_df["opponent_id"] = away_df["home_team_api_id"]
    away_df[
        "is_home_team"
    ] = 0  # Indicates this row is from the away team's perspective
    away_df["win_odds"] = away_df["away_win_odds"]
    away_df["opponent_win_odds"] = away_df["home_win_odds"]

    away_df["team_rating"] = away_df["away_team_rating"]
    away_df["opponent_team_rating"] = away_df["home_team_rating"]

    dual_perspective_df = pd.concat([home_df, away_df], ignore_index=True)

    columns_to_drop = [
        "home_team_api_id",
        "away_team_api_id",
        "home_team_goal",
        "away_team_goal",
        "home_team_rating",
        "away_team_rating",
        "home_win_odds",
        "away_win_odds",
    ]
    dual_perspective_df.drop(columns_to_drop, axis=1, inplace=True)

    # ---
    # dual_perspective_df = dual_perspective_df.sort_values(by="match_api_id")

    # For rolling stats we need to sort by Season - Stage
    dual_perspective_df = dual_perspective_df.sort_values(
        by=["season_start_year", "stage", "match_api_id"]
    )

    result_mapping = {"win": 1, "draw": 0, "loss": -1}
    dual_perspective_df["result"] = (
        dual_perspective_df["result"].map(result_mapping).astype(int)
    )

    dual_df_cp_2 = dual_perspective_df.copy()
    dual_df_cp_2.sort_values(by=["team_id", "days_after_first_date"], inplace=True)

    dual_df_cp_2["goal_deficit"] = (
            dual_df_cp_2["goals_scored"] - dual_df_cp_2["goals_conceded"]
    )

    N = 5  # Adjust N as needed

    dual_df_cp_2["points_temp"] = dual_df_cp_2["result"].map({1: 3, 0: 1, -1: 0})

    dual_df_cp_2["rolling_mean_goals_scored"] = dual_df_cp_2.groupby("team_id")[
        "goals_scored"
    ].transform(lambda x: x.rolling(window=N, min_periods=1).mean().shift())

    dual_df_cp_2["rolling_mean_goals_conceded"] = dual_df_cp_2.groupby("team_id")[
        "goals_conceded"
    ].transform(lambda x: x.rolling(window=N, min_periods=1).mean().shift())

    dual_df_cp_2["rolling_mean_goal_deficit"] = dual_df_cp_2.groupby("team_id")[
        "goal_deficit"
    ].transform(lambda x: x.rolling(window=N, min_periods=1).mean().shift())

    dual_df_cp_2["rolling_points"] = dual_df_cp_2.groupby("team_id")[
        "points_temp"
    ].transform(lambda x: x.rolling(window=N, min_periods=1).sum().shift())

    # BROKEN, TODO REMOVE:
    # dual_df_cp_2['opponent_rolling_mean_goals_scored'] = dual_df_cp_2.groupby('opponent_id')['goals_scored'].transform(
    #     lambda x: x.rolling(window=N, min_periods=1).mean().shift())
    #
    # dual_df_cp_2['opponent_rolling_mean_goals_conceded'] = dual_df_cp_2.groupby('opponent_id')['goals_conceded'].transform(
    #     lambda x: x.rolling(window=N, min_periods=1).mean().shift())
    #
    # dual_df_cp_2['opponent_rolling_mean_goal_deficit'] = dual_df_cp_2.groupby('opponent_id')['goal_deficit'].transform(
    #     lambda x: x.rolling(window=N, min_periods=1).mean().shift())
    #
    # dual_df_cp_2['opponent_rolling_points'] = dual_df_cp_2.groupby('opponent_id')['points_temp'].transform(
    #     lambda x: x.rolling(window=N, min_periods=1).sum().shift())

    # Creating a temporary DataFrame to store opponent stats
    opponent_stats = dual_df_cp_2[
        [
            "match_api_id",
            "team_id",
            "rolling_mean_goals_scored",
            "rolling_mean_goals_conceded",
            "rolling_mean_goal_deficit",
            "rolling_points",
        ]
    ].copy()

    # Renaming columns for merging
    opponent_stats.rename(
        columns={
            "team_id": "opponent_id",
            "rolling_mean_goals_scored": "opponent_rolling_mean_goals_scored",
            "rolling_mean_goals_conceded": "opponent_rolling_mean_goals_conceded",
            "rolling_mean_goal_deficit": "opponent_rolling_mean_goal_deficit",
            "rolling_points": "opponent_rolling_points",
        },
        inplace=True,
    )

    # Merging the stats back to the original DataFrame
    dual_df_cp_2 = pd.merge(
        dual_df_cp_2, opponent_stats, how="left", on=["match_api_id", "opponent_id"]
    )

    dual_df_cp_2.drop(columns=["points_temp"], inplace=True)

    dual_df = dual_df_cp_2.sort_values(by="match_api_id")

    dual_df["team_pairing"] = dual_df.apply(
        lambda x: "_".join(sorted([str(x["team_id"]), str(x["opponent_id"])])), axis=1
    )

    # Convert to categorical and then to codes
    dual_df["team_pairing_encoded"] = (
        dual_df["team_pairing"].astype("category").cat.codes
    )
    dual_df["team_rating_ratio"] = round(
        dual_df["team_rating"] / dual_df["opponent_team_rating"], 3
    )
    dual_df["capped_ratio"] = dual_df["team_rating_ratio"].clip(0.7, 1.3)
    #
    dual_df["rolling_mean_goals_scored_ratio"] = round(
        dual_df["rolling_mean_goals_scored"] / dual_df["opponent_rolling_mean_goals_scored"], 3
    )
    dual_df["rolling_mean_goals_conceded_ratio"] = round(
        dual_df["rolling_mean_goals_conceded"] / dual_df["opponent_rolling_mean_goals_conceded"], 3
    )
    dual_df["rolling_mean_goal_deficit_ratio"] = round(
        dual_df["rolling_mean_goal_deficit"] / dual_df["opponent_rolling_mean_goal_deficit"], 3
    )
    dual_df["rolling_points_ratio"] = round(
        dual_df["rolling_points"] / dual_df["opponent_rolling_points"], 3
    )

    # Optionally, drop the 'pairing' column if it's not needed
    dual_df.drop(columns=["team_pairing"], inplace=True)

    # return dual_df
    return RollingMatchTeamStats(
        dual_df=__append_season_performance(dual_df),
        player_matches_df=player_matches_df,
    )
    # return dual_df


def add_team_strategies(source_df: pd.DataFrame, team_attrs_df: pd.DataFrame):
    team_data_df_grouped = team_attrs_df.groupby("team_api_id", group_keys=False).apply(
        lambda x: x.ffill()
    )

    team_data_df_grouped = team_data_df_grouped.sort_values(
        by=["team_api_id", "days_after_first_date"]
    )
    team_data_df_grouped_renamed = team_data_df_grouped.rename(
        columns={
            col: "team_" + col
            for col in team_data_df_grouped.columns
            if col not in ["id", "team_api_id", "days_after_first_date"]
        }
    )

    source_df_copy = pd.merge_asof(
        source_df.sort_values("days_after_first_date"),
        team_data_df_grouped_renamed.rename(
            columns={
                "team_api_id": "team_id",
                "days_after_first_date": "team_days_after_first_date",
            }
        ).sort_values("team_days_after_first_date"),
        by="team_id",
        left_on="days_after_first_date",
        right_on="team_days_after_first_date",
        direction="backward",
    )

    opponent_columns_to_rename = {
        col: "opponent_" + col
        for col in team_data_df_grouped.columns
        if col not in ["id", "team_api_id", "days_after_first_date"]
    }
    team_data_df_opponent = team_data_df_grouped.rename(
        columns=opponent_columns_to_rename
    )

    source_df_copy = pd.merge_asof(
        source_df_copy.sort_values("days_after_first_date"),
        team_data_df_opponent.rename(
            columns={
                "team_api_id": "opponent_id",
                "days_after_first_date": "opp_days_after_first_date",
            }
        ).sort_values("opp_days_after_first_date"),
        by="opponent_id",
        left_on="days_after_first_date",
        right_on="opp_days_after_first_date",
        direction="backward",
        suffixes=("", "_opponent"),
    )

    source_df_copy["mean_season_points"] = (
            source_df_copy["total_season_points_before"] / source_df_copy["stage"]
    )
    source_df_copy["opponent_mean_season_points"] = (
            source_df_copy["opponent_total_season_points_before"] / source_df_copy["stage"]
    )

    source_df_copy["mean_season_points_ratio"] = round(
        source_df_copy["mean_season_points"] / source_df_copy["opponent_mean_season_points"], 3
    )


    return source_df_copy


def add_player_attrs(source_df: pd.DataFrame, player_attrs_df: pd.DataFrame):
    player_data_df_grouped = player_attrs_df.groupby("player_api_id").apply(
        lambda x: x.ffill()
    )

    player_data_df_grouped = player_data_df_grouped.sort_values(
        by=["player_api_id", "days_after_first_date"]
    )
    player_data_df_grouped_renamed = player_data_df_grouped.rename(
        columns={
            col: "player_" + col
            for col in player_data_df_grouped.columns
            if col not in ["id", "player_api_id", "days_after_first_date"]
        }
    )

    # Perform the first merge_asof with the renamed DataFrame
    source_df_copy = pd.merge_asof(
        source_df.sort_values("days_after_first_date"),
        player_data_df_grouped_renamed.rename(
            columns={
                "player_api_id": "team_id",
                "days_after_first_date": "player_days_after_first_date",
            }
        ).sort_values("player_days_after_first_date"),
        by="player_id",
        left_on="days_after_first_date",
        right_on="team_days_after_first_date",
        direction="backward",
    )

    return source_df_copy


def process_goal_info(data: LoadedDataData, full_df: pd.DataFrame):
    goals_df = data.goals_df.merge(
        data.player_df[["player_api_id", "player_name"]],
        left_on="scoring_player_id",
        right_on="player_api_id",
        how="left",
    ).drop("player_api_id", axis=1)
    return goals_df.merge(
        full_df[["match_api_id", "league_name", "team_id", "team_team_long_name"]],
        how="left",
        on=["team_id", "match_api_id"],
    )


def get_training_columns():
    return [
        # 'id_x',  # Potentially redundant as it might be an index or non-informative identifier.
        # 'days_after_first_date',
        # 'match_api_id',  # Redundant or non-informative for predictive modeling, as it's likely just an identifier.
        "result",  # Target variable or might need to be transformed depending on the model's objective.
        # 'goals_scored',
        # 'goals_conceded',
        # 'team_id',  # May need to be transformed into a feature that the model can interpret, like team statistics.
        # 'opponent_id',  # As with team_id, should be used in a way that is meaningful for the model.
        "is_home_team",
        # 'goal_deficit',
        "rolling_mean_goals_scored",
        "rolling_mean_goals_conceded",
        "rolling_mean_goal_deficit",
        # 'id_y',  # Likely a redundant identifier similar to 'id_x'.
        # 'team_team_fifa_api_id',  # Identifier, may not contribute to predictive performance.
        # REMOVE FOR NOW:
        # 'team_buildUpPlaySpeed',
        # 'team_buildUpPlaySpeedClass',
        # 'team_buildUpPlayDribbling',
        # 'team_buildUpPlayDribblingClass',
        # 'team_buildUpPlayPassing',
        # 'team_buildUpPlayPassingClass',
        # 'team_buildUpPlayPositioningClass',
        # 'team_chanceCreationPassing',
        # 'team_chanceCreationPassingClass',
        # 'team_chanceCreationCrossing',
        # 'team_chanceCreationCrossingClass',
        # 'team_chanceCreationShooting',
        # 'team_chanceCreationShootingClass',
        # 'team_chanceCreationPositioningClass',
        # 'team_defencePressure',
        # 'team_defencePressureClass',
        # 'team_defenceAggression',
        # 'team_defenceAggressionClass',
        # 'team_defenceTeamWidth',
        # 'team_defenceTeamWidthClass',
        # 'team_defenceDefenderLineClass',
        # 'team_days_after_first_date',  # May be redundant as 'days_after_first_date' already exists.
        # 'team_team_long_name',  # Might be more of a label than a feature.
        # 'team_league_name',  # Could be important if league specifics affect the game, otherwise could be an identifier.
        # 'id',  # Another identifier that might not be useful for the model.
        # 'opponent_team_fifa_api_id',  # As with 'team_team_fifa_api_id', likely an identifier.
        # REMOVE FOR NOW:
        # 'opponent_buildUpPlaySpeed',
        # 'opponent_buildUpPlaySpeedClass',
        # 'opponent_buildUpPlayDribbling',
        # 'opponent_buildUpPlayDribblingClass',
        # 'opponent_buildUpPlayPassing',
        # 'opponent_buildUpPlayPassingClass',
        # 'opponent_buildUpPlayPositioningClass',
        # 'opponent_chanceCreationPassing',
        # 'opponent_chanceCreationPassingClass',
        # 'opponent_chanceCreationCrossing',
        # 'opponent_chanceCreationCrossingClass',
        # 'opponent_chanceCreationShooting',
        # 'opponent_chanceCreationShootingClass',
        # 'opponent_chanceCreationPositioningClass',
        # 'opponent_defencePressure',
        # 'opponent_defencePressureClass',
        # 'opponent_defenceAggression',
        # 'opponent_defenceAggressionClass',
        # 'opponent_defenceTeamWidth',
        # 'opponent_defenceTeamWidthClass',
        # 'opponent_defenceDefenderLineClass',
        # 'opp_days_after_first_date',  # Similar to 'team_days_after_first_date', potentially redundant.
        # 'opponent_team_long_name',  # Likely non-informative for modeling.
        # 'opponent_league_name'  # Depends on model objective, could be important for context.
        # "stage",
        # "season_start_year",
        # "last_season_points",
        "total_season_points_before",
        "opponent_total_season_points_before",
        "team_rating",
        "opponent_team_rating"
        # "opponent_last_season_points"
    ]
