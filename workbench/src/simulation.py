import pandas as pd


def normalize_f(num, min_range=0.91, max_range=1.2):
    normalized_num = (num - min_range) / (max_range - min_range)
    return normalized_num


def kelly_criterion(win_prob, odds):
    f = win_prob - ((1 - win_prob) / odds)
    return f


def evaluate_betting_strategy(
    filtered_model_training_result,
    full_df_model,
    bankroll,
    min_range=0.91,
    max_range=1.2,
    use_kelly=False,
):
    """
    Evaluates the betting strategy based on the model's probabilities and the actual betting odds.

    Parameters:
    filtered_model_training_result: ModelTrainingResult object containing probabilities and match IDs.
    full_df_model: DataFrame with betting odds and actual results.
    bet_amount: Fixed amount to bet on each match.

    Returns:
    Total profit or loss from the betting strategy.
    """

    if use_kelly:
        bankroll = bankroll / 9

    merged_df = pd.merge(
        filtered_model_training_result.probabilities_match_id,
        full_df_model,
        on=["match_api_id", "team_id"],
    )

    merged_df_filtered = merged_df.dropna()

    total_amount_spent = 0
    total_bets = 0

    # Function to calculate profit or loss for a single match
    def calculate_pnl(row):
        # Choose the outcome to bet on (highest probability)
        predicted_outcome = row[[0, 1, 2]].idxmax(axis=0)
        odds_column = {0: "opponent_win_odds", 1: "draw_odds", 2: "win_odds"}

        predicted_outcome_prob = row[predicted_outcome]

        if use_kelly:
            f = kelly_criterion(
                predicted_outcome_prob, row[odds_column[predicted_outcome]]
            )
            normalized_f = normalize_f(f, min_range=min_range, max_range=max_range)
            bet_amount = normalized_f * bankroll

            bet_amount = max(bet_amount, 0)

            # print(
            #     f"predicted_outcome:{predicted_outcome_prob:.2f}, "
            #     f"odds: {row[odds_column[predicted_outcome]]:.2f}, "
            #     f"f: {f:.2f}, "
            #     f"normalized_f: {normalized_f:.2f}, "
            #     f"bet_amount: {bet_amount:.2f}"
            # )

        else:
            bet_amount = bankroll / len(merged_df_filtered)

        nonlocal total_bets
        total_bets += 1 if bet_amount > 0 else 0

        nonlocal total_amount_spent
        total_amount_spent += bet_amount
        if row["result"] == int(predicted_outcome):
            # Win: Profit is (Odds - 1) times the bet amount
            return (row[odds_column[predicted_outcome]] - 1) * bet_amount
        else:
            # Loss: Loss is equal to the bet amount
            return -bet_amount

    #  sum  profits/losses
    total_pnl = merged_df_filtered.apply(calculate_pnl, axis=1).sum()
    return total_pnl, total_amount_spent, total_bets, len(merged_df_filtered)


def calculate_company_profit(full_df, total_amount):
    df_filtered = full_df.dropna(subset=["draw_odds", "win_odds", "opponent_win_odds"])

    num_matches = len(df_filtered)
    bet_per_match = total_amount / num_matches

    # Calculate the compan payout
    payout_win = bet_per_match / df_filtered["win_odds"]
    payout_draw = bet_per_match / df_filtered["draw_odds"]
    payout_loss = bet_per_match / df_filtered["opponent_win_odds"]

    # the actual payout based on the result
    actual_payout = (
        (df_filtered["result"] == 1) * payout_win
        + (df_filtered["result"] == 0) * payout_draw
        + (df_filtered["result"] == -1) * payout_loss
    )

    company_earnings = bet_per_match - actual_payout

    final_balance = total_amount + company_earnings.sum()

    return final_balance
