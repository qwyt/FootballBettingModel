import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from workbench.src import feature_select
from workbench.src.preprocessors import preprocessing_for_logreg


# def preprocessing_for_logreg(extracted_features):
#     categorical_cols = extracted_features.select_dtypes(
#         include=["object", "category"]
#     ).columns
#     numerical_cols = extracted_features.select_dtypes(include=["number"]).columns
#
#     categorical_transformer = Pipeline(
#         [
#             ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore")),
#         ]
#     )
#
#     numerical_transformer = Pipeline(
#         [
#             ("imputer", SimpleImputer(strategy="mean")),
#             ("scaler", StandardScaler()),
#         ]
#     )
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numerical_transformer, numerical_cols),
#             ("cat", categorical_transformer, categorical_cols),
#         ]
#     )
#
#     return preprocessor


def preprocessing_for_svm(extracted_features):
    categorical_cols = extracted_features.select_dtypes(
        include=["object", "category"]
    ).columns
    numerical_cols = extracted_features.select_dtypes(include=["number"]).columns

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),  # Feature scaling is crucial for SVM
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


class HomeAdvantageModel:
    def fit(self, X, y):
        self.classes_ = [0, 1, 2]
        # No fitting process required for this model

    def predict(self, X):
        # Predicts 2 (win) if home team is playing, otherwise 0
        return np.where(X["is_home_team"] == True, 2, 0)

    def predict_proba(self, X):
        # Probability for each class: 0, 1, 2
        # If home team is playing, high probability for class 2, otherwise high for class 0
        proba_class_0 = np.where(X["is_home_team"] == True, 0, 1)
        proba_class_1 = np.zeros(X.shape[0])
        proba_class_2 = np.where(X["is_home_team"] == True, 1, 0)
        return np.vstack((proba_class_0, proba_class_1, proba_class_2)).T


class BettingOddsModel:
    def fit(self, X, y):
        self.classes_ = [0, 1, 2]  # 0: Loss, 0: Draw, 2: Win
        return self

    def odds_to_proba(self, odds):
        return 1 / odds

    def predict(self, X):
        win_proba = self.odds_to_proba(X["win_odds"])
        loss_proba = self.odds_to_proba(X["opponent_win_odds"])
        draw_proba = self.odds_to_proba(X["draw_odds"])

        all_proba = np.vstack((loss_proba, draw_proba, win_proba)).T
        return np.argmax(all_proba, axis=1)

    def predict_proba(self, X):
        win_proba = self.odds_to_proba(X["win_odds"])
        loss_proba = self.odds_to_proba(X["opponent_win_odds"])
        draw_proba = self.odds_to_proba(X["draw_odds"])

        total_proba = win_proba + loss_proba + draw_proba
        normalized_proba = np.vstack((loss_proba, draw_proba, win_proba)).T

        return normalized_proba


feature_set__map = {
    "Naive | Home Advantage": {
        "feature_set": feature_select.FeatureSet.Base,
        "synthetic_funcs": [],
        "model": HomeAdvantageModel,
        "supports_nan": True,
        "preprocessing": None,
        "best_params": {},
        "param_grid": {},
    },
    "Naive | Average Betting Odds": {
        "feature_set": feature_select.FeatureSet.AverageBettingOdds,
        "synthetic_funcs": [],
        "model": BettingOddsModel,
        "supports_nan": False,
        "preprocessing": None,
        "best_params": {},
        "param_grid": {},
    },
    "Logistic | Team Rating": {
        "feature_set": feature_select.FeatureSet.TeamRatingRatio,
        "synthetic_funcs": [],
        "model": LogisticRegression,
        "supports_nan": False,
        "preprocessing": preprocessing_for_logreg(),
        "best_params": {},
        "param_grid": {},
    },
    "Logistic | Team Rating + Home": {
        "feature_set": feature_select.FeatureSet.TeamRatingRatio
                       | feature_select.FeatureSet.Base,
        "synthetic_funcs": [],
        "model": LogisticRegression,
        "supports_nan": False,
        "preprocessing": preprocessing_for_logreg(),
        "best_params": {},
        "param_grid": {},
    },
    "Logistic | Full Ratios": {
        "feature_set": feature_select.FeatureSet.TeamRatingRatio
                       | feature_select.FeatureSet.Base | feature_select.FeatureSet.TeamSeasonStatsRatios,
        "synthetic_funcs": [],
        "model": LogisticRegression,
        "supports_nan": False,
        "preprocessing": preprocessing_for_logreg(),
        "best_params": {},
        "param_grid": {},
    },
    "Logistic + ElasticNet | Full Ratios": {
        "feature_set": feature_select.FeatureSet.TeamRatingRatio
                       | feature_select.FeatureSet.Base | feature_select.FeatureSet.TeamSeasonStatsRatios,
        "synthetic_funcs": [],
        "model": LogisticRegression,
        "supports_nan": False,
        "preprocessing": preprocessing_for_logreg(), 
        "best_params": dict(penalty="elasticnet", C=1.0, solver="saga", l1_ratio=0.5),
        "param_grid": {},
    },
    "Baseline |XGBoost": {
        "feature_set": feature_select.FeatureSet.Base
                       | feature_select.FeatureSet.TeamSeasonStats
                       | feature_select.FeatureSet.TeamRatingStats,
        "synthetic_funcs": [],
        "model": XGBClassifier,
        "supports_nan": True,
        "preprocessing": None,
        "best_params": {},
        "param_grid": {
            "xgb__learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
            "xgb__max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        },
    },
    "Baseline |XGBoost + Avg. Odds": {
        "feature_set": feature_select.FeatureSet.Base
                       | feature_select.FeatureSet.TeamSeasonStats
                       | feature_select.FeatureSet.TeamRatingStats
                       | feature_select.FeatureSet.AverageBettingOdds,
        "synthetic_funcs": [],
        "model": XGBClassifier,
        "supports_nan": True,
        "preprocessing": None,
        "best_params": {},
        "param_grid": {
            "xgb__learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
            "xgb__max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        },
    },
    "Baseline | Logistic Regression": {
        "feature_set": feature_select.FeatureSet.Base
                       | feature_select.FeatureSet.TeamSeasonStats
                       | feature_select.FeatureSet.TeamRatingStats,
        "synthetic_funcs": [],
        "model": LogisticRegression,
        "supports_nan": True,
        "use_cuml": False,
        "preprocessing": preprocessing_for_logreg(),
        "best_params": {},
        "param_grid": {
            "model__C": [0.1, 1, 10, 100],
            "model__solver": ["lbfgs", "saga"],
        },
    },
    "Baseline | SVM": {
        "feature_set": feature_select.FeatureSet.Base
                       | feature_select.FeatureSet.TeamSeasonStats
                       | feature_select.FeatureSet.TeamRatingStats,
        "synthetic_funcs": [],
        "model": SVC,
        "supports_nan": True,
        "use_cuml": False,
        "preprocessing": [preprocessing_for_svm],  # Adjust if needed
        "best_params": {},
        "param_grid": {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf", "poly"],
        },
    },
}


def get_config():
    config_a = [
        "Naive | Home Advantage",
        "Naive | Average Betting Odds",
        "Logistic | Team Rating",
        "Logistic | Team Rating + Home",
        # "Logistic + ElasticNet | Team Rating + Home",
        "Logistic | Full Ratios",
        # "Logistic + ElasticNet | Full Ratios",
        "Baseline |XGBoost",
        # "Baseline |XGBoost + Avg. Odds",
    ]

    config = config_a
    return {k: v for k, v in feature_set__map.items() if k in config}
