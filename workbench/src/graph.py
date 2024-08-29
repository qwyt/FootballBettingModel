from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from sklearn.metrics import confusion_matrix, auc
from workbench.src.validation import filter_matches_above_threshold
from sklearn.metrics import roc_curve
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import decomposition


# from src.pipeline import ModelTrainingResult


def component_scatter_plot(
        df,
        target,
        title,
        prefix="C",
        legend_title=None,
        layer=None,
        legend_labels=None,
        ax=None,
):
    df = df.copy()
    if ax is None:
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 8))

    palette = sns.color_palette("tab10", n_colors=df[target].nunique())
    color_mapping = {
        class_: color for class_, color in zip(df[target].unique(), palette)
    }

    if layer:
        _layer = df[df[layer] == True]
        df = df[df[layer] == False]
        layer = _layer
    sns.scatterplot(
        x=f"{prefix}1",
        y=f"{prefix}2",
        hue=target,
        data=df,
        palette=color_mapping,
        alpha=0.7 if layer is None else 0.1,
        legend=True if layer is None else False,
        # size=size,
        ax=ax,
    )

    if layer is not None:
        sns.scatterplot(
            data=layer,
            x=f"{prefix}1",
            y=f"{prefix}2",
            alpha=0.9,
            hue=target,
            legend=True,
            palette=color_mapping,
            s=100,
            ax=ax,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if legend_title:
        ax.legend(title=legend_title)
    ax.grid(True)

    if legend_labels:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [legend_labels.get(label, label) for label in labels]
        new_handles = [
            handle for label, handle in zip(labels, handles) if label in legend_labels
        ]
        ax.legend(handles=new_handles, labels=new_labels, title="Quality")
    return ax


def confusion_matrix_plot(
        model_info,
        title=None,
        axis_label: Optional[str] = None,
        labels: Optional[List[str]] = None,
        ax=None,
        annotations: str = None,
        include_sample_count=False,
        cbar=False,
        subtitle=None,
):
    if ax is None:
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 8))

    conf_matrix = confusion_matrix(model_info.y_test, model_info.predictions)
    conf_matrix_percentage = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    )

    if labels is None:
        labels = np.unique(model_info.y_test)

    # Prepare annotations for heatmap
    if include_sample_count:
        annot = np.array(
            [
                f"{pct:.2%}\n(n={count})"
                for pct, count in np.nditer([conf_matrix_percentage, conf_matrix])
            ]
        )
        annot = annot.reshape(conf_matrix.shape)
    else:
        annot = True  # Or use .2% format if you prefer

    sns.heatmap(
        conf_matrix_percentage,
        annot=annot,
        fmt="" if include_sample_count else ".2%",
        cmap=sns.diverging_palette(220, 20, as_cmap=True).reversed(),
        cbar=cbar,
        ax=ax,
    )
    ax.set_xlabel(f"Predicted Labels{'' if axis_label is None else f' ({axis_label})'}")
    ax.set_ylabel(f"True Labels{'' if axis_label is None else f' ({axis_label})'}")

    if title and subtitle:
        ax.set_title(title, pad=35)
        ax.text(
            0.5,
            1.055,
            f"(Model: {subtitle})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize="medium",
        )
    else:
        ax.set_title(title if title else "Confusion Matrix with Percentage Accuracy")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    if annotations:
        ax.text(
            0.5, -0.1, annotations, ha="center", va="center", transform=ax.transAxes
        )

    return ax


def feature_importance_plot(model, ax=None, title="Feature Importances"):
    features = model.feature_importances.index
    importances = model.feature_importances.values

    if ax is None:
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 6))

    plt.barh(features, importances, color="skyblue")
    plt.xlabel("Importance")
    plt.title(title)
    plt.gca().invert_yaxis()  # To display the highest importance at the top
    return ax
    # %%


def prediction_plot(
        model_info
        # ModelTrainingResult
        ,
        metrics: dict,
        title: str,
):
    errors = np.abs(
        model_info.y_test - model_info.predictions
    )  # Calculate absolute errors

    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        x=model_info.y_test,
        y=model_info.predictions,
        size=errors,
        alpha=0.5,
        palette="viridis",
    )

    plt.plot(
        [model_info.y_test.min(), model_info.y_test.max()],
        [model_info.y_test.min(), model_info.y_test.max()],
        "k--",
        lw=1,
        alpha=0.5,
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted Values with Error Magnitude\n{title}")
    plt.legend(title="Error Magnitude")

    # Annotation with metrics from the dictionary
    annotation_text = (
        f"MSE: {metrics['MSE']:.3f}\n"
        f"MAE: {metrics['MAE']:.3f}\n"
        f"R2: {metrics['R2']:.3f}"
    )

    plt.text(
        x=0.95,
        y=0.1,
        s=annotation_text,
        ha="right",
        va="center",
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white"),
    )

    plt.show()


def plot_threshold_metrics(
        model_training_result, min_threshold, max_threshold, model_name=None
):
    """
    Plots accuracy, F1 score, and sample count against various threshold values.

    Parameters:
    model_training_result: Result from the model training.
    min_threshold: Minimum threshold value.
    max_threshold: Maximum threshold value.
    """

    # Threshold values
    T_values = np.arange(min_threshold, max_threshold, 0.05)

    # Initialize lists to store metrics
    accuracies = []
    f1_scores = []
    logs_loss_scores = []
    sample_counts = []
    t_values_included = []

    # Calculate metrics for each threshold
    for T in T_values:
        try:
            filtered_result = filter_matches_above_threshold(model_training_result, T)
            accuracy = filtered_result.metrics["accuracy"]
            f1_score = filtered_result.metrics["f1"]
            logs_loss = filtered_result.metrics["log_loss"]
            sample_count = len(filtered_result.y_test)

            logs_loss_scores.append(logs_loss if logs_loss < 2 else 2)
            accuracies.append(accuracy)
            f1_scores.append(f1_score)
            sample_counts.append(sample_count)
            t_values_included.append(T)
        except:
            # In case there are no predictions above a given threshold (e.g. using the betting odds model)
            break
    # Creating the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Width, Height in inches

    # Plotting accuracy and F1 score on the left Y-axis
    # Plotting accuracy and F1 score on the left Y-axis
    (line1,) = ax1.plot(t_values_included, accuracies, label="Accuracy")
    (line2,) = ax1.plot(t_values_included, f1_scores, label="F1 Score")
    ax1.set_xlabel("Threshold (T)")
    ax1.set_ylabel("Performance")
    ax1.set_yticks([0.15, 0.4, 0.5, 0.6, 1])
    ax1.set_yticklabels(
        [
            "F1/Accu.",
            0.4,
            0.5,
            0.6,
            1,
        ]
    )
    ax1.set_ylim([0, 1.05])

    # ax1.set_yticks([0, .2, .4, .5, .6, .8, .9, 1, ])
    ax1.set_xticks([0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1])

    divider = make_axes_locatable(ax1)
    # ax3 = divider.append_axes("left", size="20%", pad=0.1, sharex=ax1)
    # line3, = ax1.plot(T_values, logs_loss_scores, label='Log Loss')
    ax3 = ax1.twinx()
    # ax3.spines["right"].set_position(("axes", -0.05))  # Offset the right spine of ax3
    # ax3.spines["left"].set_visible(True)
    # ax3.spines["left"].set_position(("axes", -0.1))  # Adjust to position ax3 to the right of ax1 ticks
    # ax3.yaxis.set_label_position('left')
    # ax3.yaxis.tick_left()

    # ax3.spines["left"].set_position(("axes", -0.15))  # Negative moves it leftward, away from ax1
    # ax3.spines["left"].set_visible(True)
    # ax3.yaxis.set_label_position('left')
    # ax3.yaxis.tick_left()

    # Shift ax1 ticks to the left
    ax1.spines["left"].set_position(("axes", -0.1))

    ax3.spines["left"].set_visible(True)
    ax3.spines["left"].set_position(("axes", 0.0))  # Adjust for spacing
    ax3.yaxis.set_label_position("left")
    ax3.yaxis.tick_left()

    ax3.set_ylabel("")
    # ax3.set_ylabel('Log Loss')
    ax3.plot(t_values_included, logs_loss_scores, label="Log Loss", color="purple")
    ax3.set_yticks([0.3, 0.5, 1, 1.5, 2])
    ax3.set_yticklabels(["Log Loss", 0.5, 1, 1.5, ">=2"])
    ax3.set_ylim([0, 2.1])
    ax3.grid(False)

    for _ax in [ax1, ax3]:
        for label in _ax.get_yticklabels():
            label.set_rotation(90)
            break

    (line3,) = ax3.plot(t_values_included, logs_loss_scores, label="Log Loss", color="purple")

    ax2 = ax1.twinx()
    scatter = ax2.scatter(t_values_included, sample_counts, color="green", label="Sample Count")
    ax2.set_yscale("log")
    ax2.set_ylabel("Number of Samples (log)")

    ax2.grid(False)

    ax2.set_yticks([10, 100, 1000, 5000, 25000, 50000])
    ax2.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # ax1.grid(False)
    fig.legend(
        handles=[line1, line2, line3, scatter],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
    )

    for T, count in zip(t_values_included, sample_counts):
        if count < 2500:
            ax2.annotate(
                f"{count}",
                (T, count),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    # plt.title('Metrics vs Threshold'

    title = "Model Performance By Prob. Threshold"
    if model_name:
        plt.title(title, pad=35)
        plt.text(
            0.5,
            1.055,
            f"(Model: {model_name})",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize="medium",
        )
    else:
        plt.title("Model Performance By Prob. Threshold")

    plt.tight_layout()  # Adjust layout

    plt.show()


# def render_brier_scores(dd: pd.DataFrame):
def render_by_league_beanchmark_scores(
        dd: pd.DataFrame, metric="log_loss", model_name=None
):
    dd = dd.sort_values(by="league_name")

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = np.arange(len(dd))

    odds_metric = f"odds_{metric}"
    model_metric = f"model_{metric}"

    metric_label = metric.replace("_", " ").title()

    ax.bar(
        positions - 0.15,
        dd[model_metric],
        width=0.3,
        label=f"Model {metric_label}",
        # color='blue',
        align="center",
    )
    ax.bar(
        positions + 0.15,
        dd[odds_metric],
        width=0.3,
        label=f"Odds {metric_label}",
        # color='green',
        align="center",
    )

    for pos in positions:
        ax.text(
            pos - 0.15,
            0.25,
            f"accu. = {dd['model_accuracy'][pos]:.2f}",
            ha="center",
            rotation=90,
        )
        ax.text(
            pos + 0.15,
            0.25,
            f"             {dd['odds_accuracy'][pos]:.2f}",
            ha="center",
            rotation=90,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(dd["league_name"], rotation=45, ha="right")

    ax.set_ylabel(metric_label)

    title = f"Comparison of Model and {metric_label} Scores with Accuracy Overlay"

    if model_name:
        ax.set_title(title, pad=35)
        ax.text(
            0.5,
            1.055,
            f"(Model: {model_name})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize="medium",
        )
    else:
        ax.set_title(title)

    ax.set_ylim(0, 1.5)

    ax.text(
        0.5,
        -0.5,
        "Lower score is better",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    ax.legend(loc="upper center", ncol=2)

    plt.show()


def render_dist_kurtosis_swarm_plot(league_last_games: pd.DataFrame):
    # league_gini = league_last_games.groupby('league_name')['normalized_points'].apply(gini_coefficient)
    league_kurtosis = league_last_games.groupby("league_name")[
        "normalized_points"
    ].apply(lambda x: kurtosis(x, fisher=False))

    plt.figure(figsize=(10, 6))
    ax1 = sns.swarmplot(
        data=league_last_games,
        x="normalized_points",
        hue="won_season",
        y="league_name",
        size=5,
        dodge=True,
    )
    ax1.set_title("Normalized Points and Kurtosis Across Leagues and Seasons")
    ax1.set_xlabel("Normalized Points")
    ax1.set_ylabel("League Name")

    ax1.legend(
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.3),
    )

    for t, l in zip(ax1.get_legend().texts, ["Other Teams", "Winners"]):
        t.set_text(l)

    ax1.text(
        0.5,
        -0.155,
        "Total points scored by teams between 2008 - 2015 during the season, relative to total available points",
        ha="center",
        va="center",
        fontsize=13,
        transform=ax1.transAxes,
    )

    ax2 = ax1.twinx()
    sns.lineplot(
        x=[1] * len(league_kurtosis),
        y=league_kurtosis.index,
        markers=True,
        ax=ax2,
        legend=False,
    )
    ax2.set_yticks(range(len(league_kurtosis)))
    ax2.set_yticklabels(round(league_kurtosis, 2))
    ax2.set_ylabel("Kurtosis")

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    plt.xlim(0, 1)

    plt.show()


def render_dist_swarm_plot(league_last_games: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    ax = sns.swarmplot(
        data=league_last_games, x="normalized_points", hue="won_season", y="league_name"
    )
    plt.title("Normalized Points Distribution Across Leagues and Seasons")
    plt.xlabel("Normalized Points")
    plt.ylabel("League Name")
    ax.text(
        0.5,
        -0.155,
        "Total points scored by teams between 2008 - 2015 during the season, relative to total available points",
        ha="center",
        va="center",
        fontsize=13,
        transform=ax.transAxes,
    )
    ax.legend(
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.3),
    )

    for t, l in zip(ax.get_legend().texts, ["Other Teams", "Winners"]):
        t.set_text(l)
    plt.xlim(0, 1)

    plt.show()


def render_dist_plot(league_last_games_for_ratings, type="Violin"):
    plt.figure(figsize=(12, 8))

    if type == "Violin":
        ax = sns.violinplot(
            data=league_last_games_for_ratings,
            x="league_name",
            y="avg_season_team_rating",
            hue="top_4",
            split=True,
            inner="quart",
            fill=False,
            # palette={"Yes": "g", "No": ".35"}
        )
    elif type == "Box":
        ax = sns.boxplot(
            data=league_last_games_for_ratings,
            x="league_name",
            y="avg_season_team_rating",
            # hue="top_4",
            # split=True,
            # inner="quart",
            # fill=False,
            # palette={"Yes": "g", "No": ".35"}
        )
    plt.xticks(rotation=45)
    ax.legend(
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.015),
    )
    for t, l in zip(ax.get_legend().texts, ["Other Teams", "Top 4 Teams"]):
        t.set_text(l)

    ax.set_title("Distribution of Team player Ratings by League")


def roc_curve_plot(y_true, y_probs, ax, labels, annotations, class_idx, n):
    """
    Plot ROC curve for a specific class in a multi-class classification.

    Args:
    y_true (pd.Series): True labels.
    y_probs (pd.Series): Probability predictions for the specific class.
    ax (matplotlib.axes.Axes): Axes object to plot on.
    labels (list): Class labels.
    annotations (str): Text for annotations.
    class_idx (int): Index of the class to plot ROC for.
    n (int): Number of rows.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot
    ax.plot(fpr, tpr, label=f"Class {labels[class_idx]} (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: Class {labels[class_idx]}")
    ax.legend(loc="lower right")
    annotations += f", n={n}"
    # ax.text(0.5, 0.2, annotations, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, -0.2, annotations, ha="center", va="center", transform=ax.transAxes)


def render_feature_importances_chart(
        feature_importances,
        title="Feature Importance",
        subtitle=None,
):
    # Feature Importance Plot

    # feature_importances = benchmark_model_target_xgb.feature_importances_
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=feature_importances, y="Feature", x="Importance")
    ax.tick_params(axis="x", which="major", labelsize=16)

    def custom_label_function(original_label):
        modified_label = original_label.replace("_", " ")
        modified_label = modified_label.replace("league name", " ")
        return modified_label.title()

    ax.set_yticklabels(
        [
            custom_label_function(label)
            for label in [item.get_text() for item in ax.get_yticklabels()]
        ]
    )

    if title and subtitle:
        plt.title(title, pad=30)
        plt.text(
            0.5,
            1.035,
            f"(Model: {subtitle})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize="medium",
        )
    else:
        plt.title(title if title else "Confusion Matrix with Percentage Accuracy")

    plt.show()


def render_pca_component_plot(_explained_pc, title=None):
    plt.figure(figsize=(12, 6))
    ax_b = barplot = sns.barplot(
        x="PC",
        y="var",
        data=_explained_pc,
        color="skyblue",
        label="Individual Variance",
        legend=False  # Disable the

    )
    ax_b.set_ylim([0, 1.1])  # Adjust as needed

    barplot.set_ylim([0, 0.5])  # Adjust as needed
    # barplot.legend_.remove()

    lineplot_ax = plt.twinx()

    lineplot = sns.lineplot(
        x="PC",
        y="cum_var",
        data=_explained_pc,
        marker="o",
        sort=False,
        color="red",
        label="Cumulative Variance",
        ax=lineplot_ax,
    )
    ax_b.grid(False)
    lineplot_ax.set_yticks([0, .2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    ax_b.set_yticks([0, .1, .2, .3, .4, .5])

    barplot.set_ylabel("Individual Variance")
    lineplot_ax.set_ylabel("Cumulative Variance")
    plt.title(title if title else "PCA Explained Variance and Cumulative Variance")
    handles, labels = [], []

    for ax in [barplot, lineplot_ax]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    plt.legend(handles, labels, loc="center right")

    plt.show()


def render_goal_type_plot(pivot_df, combined_df, goal_type_legend_labels):
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    pivot_df.plot(kind="barh", stacked=True, ax=ax1)

    ax2 = ax1.twinx()
    assist_scatter = ax2.scatter(
        combined_df["assist_proportion"],
        combined_df["league_name"],
        marker="|",
        color="black",
    )
    ax2.set_yticklabels([])
    ax2.grid(False)

    handles, labels = ax1.get_legend_handles_labels()
    custom_labels = [goal_type_legend_labels.get(label, label) for label in labels]

    assist_legend = plt.Line2D(
        [0], [0], marker="|", color="black", label="Assist", markersize=10
    )

    # Combine custom legend entries
    handles.append(assist_legend)
    custom_labels.append("% assists")

    ax1.legend(
        handles,
        custom_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=len(handles),
    )

    ax1.set_xlabel("Proportion")
    ax1.set_ylabel("League")
    ax1.set_title("Proportion of Goal Types and Assists per League")

    plt.tight_layout()
    plt.show()


def plot_player_goal_inequality(proportions, total_players_by_league):
    fig, ax = plt.subplots(figsize=(10, 6))
    proportions.T.plot(kind="barh", stacked=True, ax=ax)

    for i, league in enumerate(proportions.columns):
        total_players = total_players_by_league[league]
        ax.text(1.01, i, f"n={total_players}", va="center")

    ax.set_xlabel("Percentage of Goals")
    ax.set_ylabel("League")
    ax.set_title("Proportion of Goals by Player Categories in Each League")
    ax.text(
        0.5,
        -2.05,
        f"*Poland and Scotland excluded due to missing data",
        fontsize="medium",
        ha="right",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.095),
        ncol=len(proportions.columns),
        frameon=False,
    )

    plt.show()
