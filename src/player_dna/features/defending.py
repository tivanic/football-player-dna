from __future__ import annotations

import numpy as np
import pandas as pd


SUCCESS_OUTCOMES = {
    "Won",
    "Success In Play",
    "Success Out",
}


def _boolean_column(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a boolean column, treating a missing column/value as False."""

    if column not in df.columns:
        return pd.Series(False, index=df.index)

    return df[column].fillna(False).astype(bool)


def _string_column(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a string-like column even if it is absent."""

    if column not in df.columns:
        return pd.Series(
            pd.NA,
            index=df.index,
            dtype="object",
        )

    return df[column]


def build_defending_features(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build match-level defensive features for every player."""

    group_columns = [
        "player.id",
        "player.name",
        "team.name",
    ]

    players = (
        events
        .dropna(
            subset=[
                "player.id",
                "player.name",
                "team.name",
            ]
        )
        [group_columns]
        .drop_duplicates()
        .copy()
    )

    # --------------------------------------------------
    # Pressures
    # --------------------------------------------------

    pressures = (
        events[
            events["type.name"] == "Pressure"
        ]
        .dropna(subset=group_columns)
        .groupby(group_columns)
        .size()
        .reset_index(name="pressures")
    )

    # --------------------------------------------------
    # Ball recoveries
    # --------------------------------------------------

    recoveries = events[
        events["type.name"] == "Ball Recovery"
    ].dropna(
        subset=group_columns
    ).copy()

    recoveries["recovery_failure"] = _boolean_column(
        recoveries,
        "ball_recovery.recovery_failure",
    )

    recoveries["successful_recovery"] = (
        ~recoveries["recovery_failure"]
    )

    recoveries["offensive_recovery"] = _boolean_column(
        recoveries,
        "ball_recovery.offensive",
    )

    recovery_features = (
        recoveries
        .groupby(group_columns)
        .agg(
            ball_recoveries=("id", "size"),
            successful_recoveries=(
                "successful_recovery",
                "sum",
            ),
            offensive_recoveries=(
                "offensive_recovery",
                "sum",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Duels / tackles
    # --------------------------------------------------

    duels = events[
        events["type.name"] == "Duel"
    ].dropna(
        subset=group_columns
    ).copy()

    duel_type = _string_column(
        duels,
        "duel.type.name",
    )

    duel_outcome = _string_column(
        duels,
        "duel.outcome.name",
    )

    tackles = duels[
        duel_type == "Tackle"
    ].copy()

    tackle_outcome = _string_column(
        tackles,
        "duel.outcome.name",
    )

    tackles["successful_tackle"] = (
        tackle_outcome.isin(SUCCESS_OUTCOMES)
    )

    tackle_features = (
        tackles
        .groupby(group_columns)
        .agg(
            tackles=("id", "size"),
            successful_tackles=(
                "successful_tackle",
                "sum",
            ),
        )
        .reset_index()
    )

    aerial_lost = (
        duels[
            duel_type == "Aerial Lost"
        ]
        .groupby(group_columns)
        .size()
        .reset_index(name="aerial_duels_lost")
    )

    # --------------------------------------------------
    # Interceptions
    # --------------------------------------------------

    interceptions = events[
        events["type.name"] == "Interception"
    ].dropna(
        subset=group_columns
    ).copy()

    interception_outcome = _string_column(
        interceptions,
        "interception.outcome.name",
    )

    interceptions["successful_interception"] = (
        interception_outcome.isin(SUCCESS_OUTCOMES)
    )

    interception_features = (
        interceptions
        .groupby(group_columns)
        .agg(
            interceptions=("id", "size"),
            successful_interceptions=(
                "successful_interception",
                "sum",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Blocks
    # --------------------------------------------------

    blocks = (
        events[
            events["type.name"] == "Block"
        ]
        .dropna(subset=group_columns)
        .groupby(group_columns)
        .size()
        .reset_index(name="blocks")
    )

    # --------------------------------------------------
    # Dribbled past
    # --------------------------------------------------

    dribbled_past = (
        events[
            events["type.name"] == "Dribbled Past"
        ]
        .dropna(subset=group_columns)
        .groupby(group_columns)
        .size()
        .reset_index(name="dribbled_past")
    )

    # --------------------------------------------------
    # Fouls committed
    # --------------------------------------------------

    fouls = events[
        events["type.name"] == "Foul Committed"
    ].dropna(
        subset=group_columns
    ).copy()

    card_name = _string_column(
        fouls,
        "foul_committed.card.name",
    )

    fouls["yellow_card"] = (
        card_name == "Yellow Card"
    )

    fouls["red_card"] = (
        card_name.isin(
            {
                "Red Card",
                "Second Yellow",
            }
        )
    )

    foul_features = (
        fouls
        .groupby(group_columns)
        .agg(
            fouls_committed=("id", "size"),
            yellow_cards=("yellow_card", "sum"),
            red_cards=("red_card", "sum"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------

    features = players

    for feature_table in [
        pressures,
        recovery_features,
        tackle_features,
        aerial_lost,
        interception_features,
        blocks,
        dribbled_past,
        foul_features,
    ]:
        features = features.merge(
            feature_table,
            on=group_columns,
            how="left",
        )

    count_columns = [
        "pressures",
        "ball_recoveries",
        "successful_recoveries",
        "offensive_recoveries",
        "tackles",
        "successful_tackles",
        "aerial_duels_lost",
        "interceptions",
        "successful_interceptions",
        "blocks",
        "dribbled_past",
        "fouls_committed",
        "yellow_cards",
        "red_cards",
    ]

    features[count_columns] = (
        features[count_columns]
        .fillna(0)
    )

    features["tackle_success_pct"] = np.where(
        features["tackles"] > 0,
        (
            features["successful_tackles"]
            / features["tackles"]
            * 100
        ),
        np.nan,
    )

    features["interception_success_pct"] = np.where(
        features["interceptions"] > 0,
        (
            features["successful_interceptions"]
            / features["interceptions"]
            * 100
        ),
        np.nan,
    )

    return features