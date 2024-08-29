import numpy as np
import pandas as pd

# from cuml.metrics import confusion_matrix
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
)
from workbench.src.shared import ModelTrainingResult


def calc_metrics(predictions, probs_a, y_test):
    metrics = {}
    metrics["f1"] = f1_score(y_test, predictions, average="macro")
    metrics["accuracy"] = accuracy_score(y_test, predictions)
    metrics["precision"] = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    metrics["recall"] = recall_score(y_test, predictions, average="macro")
    metrics["log_loss"] = log_loss(y_test, probs_a)

    metrics = {key: round(metrics[key], 4) for key in metrics}
    return metrics


def evaluate_model(model, X_test, y_test, x_test_match_ids=None, continuous=False):
    predictions = model.predict(X_test)
    predictions = pd.Series(predictions, index=X_test.index)

    probabilities = model.predict_proba(X_test)
    probabilities = pd.DataFrame(probabilities, index=X_test.index)

    probabilities_match_id = None

    if x_test_match_ids is not None:
        if not x_test_match_ids.index.equals(probabilities.index):
            raise ValueError("Index does not match.")
        else:
            probabilities_match_id = probabilities.copy()
            probabilities_match_id["match_api_id"] = x_test_match_ids["match_api_id"]
            probabilities_match_id["team_id"] = x_test_match_ids["team_id"]

    metrics = {}

    rounded_predictions = predictions

    metrics = calc_metrics(rounded_predictions, probabilities[[0, 1, 2]], y_test)

    # Calculate Brier score
    log_loss_value = log_loss(y_test, probabilities)
    metrics["log_loss"] = log_loss_value

    return metrics, predictions, probabilities, probabilities_match_id


def compute_class_accuracies(predictions, x_test, y_test, classes):
    predictions = np.rint(predictions).astype(int)
    cm = confusion_matrix(y_test, predictions)
    class_accuracies = {}

    for idx, class_label in enumerate(classes):
        accuracy = cm[idx, idx] / cm[idx].sum()
        total_samples = cm[idx].sum()
        class_accuracies[class_label] = {
            "accuracy": accuracy,
            "total_samples": total_samples,
        }

    return class_accuracies


def _compute_class_accuracies(model, x_test, y_test):
    predictions = model.predict(x_test)
    return compute_class_accuracies(predictions, x_test, y_test, model.classes_)


def filter_matches_above_threshold(model_training_result, threshold):
    high_prob_indices = model_training_result.probabilities.max(axis=1) > threshold

    # Filter the data
    filtered_y_test = model_training_result.y_test[high_prob_indices]
    filtered_predictions = model_training_result.predictions[high_prob_indices]
    filtered_x_test = model_training_result.x_test[high_prob_indices]
    filter_probs_a = model_training_result.probabilities[high_prob_indices]

    # Update metrics and class accuracies based on filtered data
    updated_metrics = calc_metrics(
        filtered_predictions, filter_probs_a, filtered_y_test
    )
    classes = np.sort(filtered_y_test.unique())  # Assuming class labels are in y_test
    updated_class_accuracies = compute_class_accuracies(
        filtered_predictions, filtered_x_test, filtered_y_test, classes
    )

    filtered_model_training_result = ModelTrainingResult(
        y_test=filtered_y_test,
        x_test=filtered_x_test,
        predictions=filtered_predictions,
        probabilities=model_training_result.probabilities[high_prob_indices],
        probabilities_match_id=model_training_result.probabilities_match_id[
            high_prob_indices
        ],
        metrics=updated_metrics,
        class_accuracies=updated_class_accuracies,
    )

    return filtered_model_training_result


def calculate_scores_by_league(model_probs, full_df_model, by_league=False):
    # Merge model probabilities with full_df_model
    combined_df = pd.merge(model_probs, full_df_model, on=["match_api_id", "team_id"])

    # Exclude rows with NaN in betting odds
    combined_df = combined_df.dropna(
        subset=["win_odds", "draw_odds", "opponent_win_odds"]
    )

    # Convert betting odds to normalized probabilities
    combined_df["prob_win"] = 1 / combined_df["win_odds"]
    combined_df["prob_draw"] = 1 / combined_df["draw_odds"]
    combined_df["prob_loss"] = 1 / combined_df["opponent_win_odds"]
    total_prob = combined_df[["prob_win", "prob_draw", "prob_loss"]].sum(axis=1)
    combined_df[["prob_win", "prob_draw", "prob_loss"]] /= total_prob[:, None]

    # Function to calculate metrics
    def calculate_metrics(df):
        model_pred = df[[0, 1, 2]].idxmax(axis=1)

        model_probs = df[[0, 1, 2]]

        odds_probs = df[["prob_loss", "prob_draw", "prob_win", "result"]].copy()

        odds_probs = odds_probs.dropna()
        odds_probs.columns = [0, 1, 2, "result"]

        odds_actual = odds_probs["result"]
        odds_probs = odds_probs[[0, 1, 2]]
        odds_pred = odds_probs.idxmax(axis=1)

        # mapping = {"prob_loss": 0, "prob_draw": 1, "prob_win": 2}
        # odds_pred = odds_pred.map(mapping)

        # Actual class
        actual = df["result"]

        # Calculate Brier Score, Accuracy, and F1 Score
        model_brier_score = np.mean(
            (df[2] - (actual == 2).astype(int)) ** 2
            + (df[1] - (actual == 1).astype(int)) ** 2
            + (df[0] - (actual == 0).astype(int)) ** 2
        )
        odds_brier_score = np.mean(
            (df["prob_win"] - (actual == 2).astype(int)) ** 2
            + (df["prob_draw"] - (actual == 1).astype(int)) ** 2
            + (df["prob_loss"] - (actual == 0).astype(int)) ** 2
        )
        model_accuracy = accuracy_score(actual, model_pred)
        odds_accuracy = accuracy_score(odds_actual, odds_pred)
        model_f1 = f1_score(actual, model_pred, average="weighted")
        odds_f1 = f1_score(odds_actual, odds_pred, average="weighted")

        model_log_loss = log_loss(actual, model_probs)
        odds_log_loss = log_loss(odds_actual, odds_probs)

        return pd.Series(
            {
                "model_log_loss": model_log_loss,
                "odds_log_loss": odds_log_loss,
                "model_brier_score": model_brier_score,
                "odds_brier_score": odds_brier_score,
                "model_accuracy": model_accuracy,
                "odds_accuracy": odds_accuracy,
                "model_f1_score": model_f1,
                "odds_f1_score": odds_f1,
                "n": len(df),
            }
        )

    # Group by league_name and calculate metrics

    if by_league:
        results = (
            combined_df.groupby("league_name").apply(calculate_metrics).reset_index()
        )
    else:
        combined_df["m"] = 1
        results = combined_df.groupby("m").apply(calculate_metrics).reset_index()
    return results


def benchmark_model_probs(_model_probs, _df_model, by_league=True):
    model_benchmark = calculate_scores_by_league(
        _model_probs, _df_model, by_league=by_league
    )
    model_benchmark.insert(
        model_benchmark.columns.get_loc("odds_log_loss") + 1,
        "odds_log_loss_advantage",
        model_benchmark["odds_log_loss"] - model_benchmark["model_log_loss"],
    ),
    model_benchmark.insert(
        model_benchmark.columns.get_loc("odds_brier_score") + 1,
        "odds_accu_advantage",
        model_benchmark["model_accuracy"] - model_benchmark["odds_accuracy"],
    )
    model_benchmark = model_benchmark.sort_values(
        by=["odds_log_loss_advantage"], ascending=False
    )
    return model_benchmark


def calculate_brier_score(full_df):
    """
    Brier Score: A common method to assess the accuracy of probabilistic predictions.

    It's used to calculate the quality of odds prediction by betting companies and our models

    :param full_df:
    :return:
    """
    # Filter rows where odds are available
    df_filtered = full_df.dropna(
        subset=["draw_odds", "win_odds", "opponent_win_odds"]
    ).copy()

    # Convert odds to probabilities and normalize
    df_filtered["prob_win"] = 1 / df_filtered["win_odds"]
    df_filtered["prob_draw"] = 1 / df_filtered["draw_odds"]
    df_filtered["prob_loss"] = 1 / df_filtered["opponent_win_odds"]
    total_prob = df_filtered[["prob_win", "prob_draw", "prob_loss"]].sum(axis=1)
    df_filtered[["prob_win", "prob_draw", "prob_loss"]] /= total_prob[:, None]

    # Calculate Brier Score
    df_filtered["brier_score"] = (
        (df_filtered["prob_win"] - (df_filtered["result"] == 1).astype(int)) ** 2
        + (df_filtered["prob_draw"] - (df_filtered["result"] == 0).astype(int)) ** 2
        + (df_filtered["prob_loss"] - (df_filtered["result"] == -1).astype(int)) ** 2
    )

    # Average Brier Score
    average_brier_score = df_filtered["brier_score"].mean()

    return average_brier_score
