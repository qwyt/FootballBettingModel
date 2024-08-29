from unittest import TestCase

from workbench.src.data_process import add_team_strategies
import pandas as pd


class DataProcessAddTeamStrategiesTests(TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock data for source_df with 10 different games
        cls.source_df = pd.DataFrame(
            {
                "days_after_first_date": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
                "match_api_id": range(1, 11),
                "team_id": [1001] * 5 + [1002] * 5,
                "opponent_id": [
                    2001,
                    2002,
                    2003,
                    2004,
                    2005,
                    2001,
                    2002,
                    2003,
                    2004,
                    2005,
                ],
            }
        )

        # Mock data for team_attrs_df with multiple strategies and one not applicable
        cls.team_attrs_df = pd.DataFrame(
            {
                "team_api_id": [1001] * 4 + [1002] * 3 + [3001],
                "days_after_first_date": [5, 20, 30, 60, 10, 45, 60, 75],
                "strategy_attribute": ["A", "B", "C", "D", "E", "F", "G", "H"],
            }
        )

    def test_basic_functionality(self):
        result_df = add_team_strategies(self.source_df, self.team_attrs_df)
        self.assertIn("team_strategy_attribute", result_df.columns)
        self.assertIn("opponent_strategy_attribute", result_df.columns)

    def test_order_preservation(self):
        result_df = add_team_strategies(self.source_df, self.team_attrs_df)
        pd.testing.assert_frame_equal(
            result_df[["match_api_id"]],
            self.source_df[["match_api_id"]],
            check_like=True,
        )

    def test_data_integrity(self):
        result_df = add_team_strategies(self.source_df, self.team_attrs_df)
        self.assertEqual(
            result_df.loc[
                result_df["match_api_id"] == 2, "team_strategy_attribute"
            ].iloc[0],
            "A",
        )
        self.assertEqual(
            result_df.loc[
                result_df["match_api_id"] == 5, "team_strategy_attribute"
            ].iloc[0],
            "C",
        )

    def test_correct_strategy_selection(self):
        result_df = add_team_strategies(self.source_df, self.team_attrs_df)
        # Testing correct strategy selection for team 1001 and 1002
        correct_strategies = ["A", "A", "B", "B", "C", "E", "E", "F", "F", "F"]
        selected_strategies = result_df["team_strategy_attribute"].tolist()
        self.assertEqual(selected_strategies, correct_strategies)

    def test_strategy_not_applicable(self):
        result_df = add_team_strategies(self.source_df, self.team_attrs_df)
        self.assertTrue(
            pd.isna(
                result_df[result_df["team_id"] == 3001]["team_strategy_attribute"]
            ).all()
        )
