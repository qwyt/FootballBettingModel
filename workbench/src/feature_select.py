"""

"""
from enum import IntEnum
from typing import List

import numpy as np
import pandas as pd


class FeatureSet(IntEnum):
    Base = 1

    TeamIntersectionData = 2

    TeamSeasonStats = 4

    TeamStyle = 8

    TeamSeasonOptional = 16
    TeamRatingStats = 32
    AverageBettingOdds = 64
    TeamRatingRatio = 128
    TeamSeasonStatsRatios = 256


def _get_feature_set(val: FeatureSet):
    if val == FeatureSet.Base:
        return [
            "is_home_team"
            # "stage",
            # "season_start_year",
            # "team_days_after_first_date",
            # "team_league_name",
        ]
    elif val == FeatureSet.AverageBettingOdds:
        return ["draw_odds", "win_odds", "opponent_win_odds"]
    elif val == FeatureSet.TeamIntersectionData:
        return ["team_pairing_encoded"]
    elif val == FeatureSet.TeamRatingRatio:
        return ["team_rating_ratio"]
    elif val == FeatureSet.TeamRatingStats:
        return ["team_rating", "opponent_team_rating"]
    elif val == FeatureSet.TeamSeasonStats:
        return [
            "mean_season_points",
            "opponent_mean_season_points",
            "rolling_mean_goals_scored",
            "rolling_mean_goals_conceded",
            "rolling_mean_goal_deficit",
            "opponent_rolling_mean_goals_scored",
            "opponent_rolling_mean_goals_conceded",
            "opponent_rolling_mean_goal_deficit",
            "rolling_points",
            "opponent_rolling_points",
        ]
    elif val == FeatureSet.TeamSeasonStatsRatios:
        return [
            "mean_season_points_ratio",
            "rolling_mean_goals_scored_ratio",
            "rolling_mean_goals_conceded_ratio",
            "rolling_mean_goal_deficit_ratio",
            "rolling_points_ratio"
        ]

    elif val == FeatureSet.TeamSeasonOptional:
        return [
            "last_season_points",  # Missing for first year
            "opponent_last_season_points",  # Missing for first year
        ]
    elif val == FeatureSet.TeamStyle:
        return [
            "opponent_buildUpPlaySpeed",
            "opponent_buildUpPlaySpeedClass",
            "opponent_buildUpPlayDribbling",
            "opponent_buildUpPlayDribblingClass",
            "opponent_buildUpPlayPassing",
            "opponent_buildUpPlayPassingClass",
            "opponent_buildUpPlayPositioningClass",
            "opponent_chanceCreationPassing",
            "opponent_chanceCreationPassingClass",
            "opponent_chanceCreationCrossing",
            "opponent_chanceCreationCrossingClass",
            "opponent_chanceCreationShooting",
            "opponent_chanceCreationShootingClass",
            "opponent_chanceCreationPositioningClass",
            "opponent_defencePressure",
            "opponent_defencePressureClass",
            "opponent_defenceAggression",
            "opponent_defenceAggressionClass",
            "opponent_defenceTeamWidth",
            "opponent_defenceTeamWidthClass",
            "opponent_defenceDefenderLineClass",
            "team_buildUpPlaySpeed",
            "team_buildUpPlaySpeedClass",
            "team_buildUpPlayDribbling",
            "team_buildUpPlayDribblingClass",
            "team_buildUpPlayPassing",
            "team_buildUpPlayPassingClass",
            "team_buildUpPlayPositioningClass",
            "team_chanceCreationPassing",
            "team_chanceCreationPassingClass",
            "team_chanceCreationCrossing",
            "team_chanceCreationCrossingClass",
            "team_chanceCreationShooting",
            "team_chanceCreationShootingClass",
            "team_chanceCreationPositioningClass",
            "team_defencePressure",
            "team_defencePressureClass",
            "team_defenceAggression",
            "team_defenceAggressionClass",
            "team_defenceTeamWidth",
            "team_defenceTeamWidthClass",
            "team_defenceDefenderLineClass",
        ]


def get_feature_sets(val: int) -> List[str]:
    results = []
    for feature in FeatureSet:
        if val & feature:
            results.extend(_get_feature_set(feature))
    return results


# def get_interaction_vars_for_group(df: pd.DataFrame, drop_previous=True) -> pd.DataFrame:
#     # if val != FeatureSet.TeamStyle:
#     #     raise Exception("TODO")
#
#     val = FeatureSet.TeamStyle
#     features = _get_feature_set(val)
#
#     features = [f for f in features if "team_" in f]
#
#     for feature in features:
#         _feature = feature.replace("team_", "")
#         team_col = f'team_{_feature}'
#         opp_col = f'opponent_{_feature}'
#         interaction_col = f'interaction_{_feature}'
#
#         # Create interaction feature by concatenating the team and opponent feature values
#         # df[interaction_col] = df[team_col].astype(str) + '_' + df[opp_col].astype(str)
#         df[interaction_col] = df[team_col] * 111 + df[opp_col]
#
#         if drop_previous:
#             df = df.drop(columns=[team_col, opp_col])
#
#     return df
#
# def get_relative_team_stats(df: pd.DataFrame, drop_previous=True) -> pd.DataFrame:
#
#     val = FeatureSet.TeamSeasonStats
#     features = _get_feature_set(val)
#
#     features = [f for f in features if "opponent_" in f]
#
#     for feature in features:
#         _feature = feature.replace("opponent_", "")
#         team_col = f'{_feature}'
#         opp_col = f'opponent_{_feature}'
#         interaction_col = f'relative_{_feature}'
#
#         df[interaction_col] = df[team_col] / df[opp_col]
#         df[interaction_col].replace([np.inf, -np.inf], 5, inplace=True)
#         df[interaction_col].fillna(0, inplace=True)
#         df[interaction_col] = 5 * (df[interaction_col] - df[interaction_col].min()) / (
#                     df[interaction_col].max() - df[interaction_col].min())
#
#         if drop_previous:
#             df = df.drop(columns=[team_col, opp_col])
#
#     return df
#
#
# def get_relative_team_ratings(df: pd.DataFrame, drop_previous=True) -> pd.DataFrame:
#
#     val = FeatureSet.TeamRatingStats
#     features = _get_feature_set(val)
#
#     features = [f for f in features if "opponent_" in f]
#
#     for feature in features:
#         _feature = feature.replace("opponent_", "")
#         team_col = f'{_feature}'
#         opp_col = f'opponent_{_feature}'
#         interaction_col = f'relative_{_feature}'
#
#         df[interaction_col] = df[team_col] / df[opp_col]
#
#         if drop_previous:
#             df = df.drop(columns=[team_col, opp_col])
#
#     return df
def _process_features(
    df: pd.DataFrame, features: list, drop_previous: bool, normalize=False
) -> pd.DataFrame:
    for feature in features:
        _feature = feature.replace("opponent_", "")
        team_col = f"{_feature}"
        opp_col = f"opponent_{_feature}"
        interaction_col = f"relative_{_feature}"

        df[interaction_col] = df[team_col] / df[opp_col]

        if normalize:
            df[interaction_col].replace([np.inf, -np.inf], 5, inplace=True)
            df[interaction_col].fillna(0, inplace=True)
            df[interaction_col] = (
                5
                * (df[interaction_col] - df[interaction_col].min())
                / (df[interaction_col].max() - df[interaction_col].min())
            )

        if drop_previous:
            df = df.drop(columns=[team_col, opp_col])

    return df


def get_relative_team_stats(df: pd.DataFrame, drop_previous=True) -> pd.DataFrame:
    val = FeatureSet.TeamSeasonStats
    features = _get_feature_set(val)
    features = [f for f in features if "opponent_" in f]

    return _process_features(df, features, drop_previous, normalize=True)


def get_relative_team_ratings(df: pd.DataFrame, drop_previous=True) -> pd.DataFrame:
    val = FeatureSet.TeamRatingStats
    features = _get_feature_set(val)
    features = [f for f in features if "opponent_" in f]

    return _process_features(df, features, drop_previous)
