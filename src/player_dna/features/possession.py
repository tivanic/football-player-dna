from __future__ import annotations

import numpy as np
import pandas as pd


def _coordinate(value: object, index: int) -> float:
    """Safely extract x or y from a StatsBomb location."""

    if isinstance(value, list) and len(value) > index:
        return float(value[index])

    return np.nan


def build_possession_features(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build match-level possession features for every player."""

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
    # Carries
    # --------------------------------------------------

    carries = events[
        events["type.name"] == "Carry"
    ].dropna(
        subset=[
            "player.id",
            "player.name",
            "team.name",
        ]
    ).copy()

    carries["start_x"] = carries["location"].apply(
        lambda value: _coordinate(value, 0)
    )
    carries["start_y"] = carries["location"].apply(
        lambda value: _coordinate(value, 1)
    )

    carries["end_x"] = carries[
        "carry.end_location"
    ].apply(
        lambda value: _coordinate(value, 0)
    )
    carries["end_y"] = carries[
        "carry.end_location"
    ].apply(
        lambda value: _coordinate(value, 1)
    )

    carries["x_progression"] = (
        carries["end_x"] - carries["start_x"]
    )

    carries["carry_length"] = np.hypot(
        carries["end_x"] - carries["start_x"],
        carries["end_y"] - carries["start_y"],
    )

    # Baseline project definition:
    # a carry moving the ball at least 10 StatsBomb
    # units toward the opponent goal.
    carries["progressive_carry"] = (
        carries["x_progression"] >= 10
    )

    carries["final_third_entry"] = (
        (carries["start_x"] < 80)
        & (carries["end_x"] >= 80)
    )

    start_in_box = (
        (carries["start_x"] >= 102)
        & carries["start_y"].between(18, 62)
    )

    end_in_box = (
        (carries["end_x"] >= 102)
        & carries["end_y"].between(18, 62)
    )

    carries["carry_into_box"] = (
        ~start_in_box
        & end_in_box
    )

    carry_features = (
        carries
        .groupby(group_columns)
        .agg(
            carries=("id", "size"),
            total_carry_distance=("carry_length", "sum"),
            average_carry_length=("carry_length", "mean"),
            progressive_carries=("progressive_carry", "sum"),
            carries_into_final_third=("final_third_entry", "sum"),
            carries_into_box=("carry_into_box", "sum"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Dribbles
    # --------------------------------------------------

    dribbles = events[
        events["type.name"] == "Dribble"
    ].dropna(
        subset=[
            "player.id",
            "player.name",
            "team.name",
        ]
    ).copy()

    dribbles["successful_dribble"] = (
        dribbles["dribble.outcome.name"]
        == "Complete"
    )

    dribble_features = (
        dribbles
        .groupby(group_columns)
        .agg(
            dribbles=("id", "size"),
            successful_dribbles=(
                "successful_dribble",
                "sum",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Ball receipts
    # --------------------------------------------------

    receipts = events[
        events["type.name"] == "Ball Receipt*"
    ].dropna(
        subset=[
            "player.id",
            "player.name",
            "team.name",
        ]
    ).copy()

    receipts["incomplete_receipt"] = (
        receipts["ball_receipt.outcome.name"]
        == "Incomplete"
    )

    receipt_features = (
        receipts
        .groupby(group_columns)
        .agg(
            ball_receipts=("id", "size"),
            incomplete_ball_receipts=(
                "incomplete_receipt",
                "sum",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Dispossessed
    # --------------------------------------------------

    dispossessed = (
        events[
            events["type.name"] == "Dispossessed"
        ]
        .dropna(
            subset=[
                "player.id",
                "player.name",
                "team.name",
            ]
        )
        .groupby(group_columns)
        .size()
        .reset_index(name="dispossessed")
    )

    # --------------------------------------------------
    # Miscontrols
    # --------------------------------------------------

    miscontrols = (
        events[
            events["type.name"] == "Miscontrol"
        ]
        .dropna(
            subset=[
                "player.id",
                "player.name",
                "team.name",
            ]
        )
        .groupby(group_columns)
        .size()
        .reset_index(name="miscontrols")
    )

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------

    features = players

    for feature_table in [
        carry_features,
        dribble_features,
        receipt_features,
        dispossessed,
        miscontrols,
    ]:
        features = features.merge(
            feature_table,
            on=group_columns,
            how="left",
        )

    count_columns = [
        "carries",
        "progressive_carries",
        "carries_into_final_third",
        "carries_into_box",
        "dribbles",
        "successful_dribbles",
        "ball_receipts",
        "incomplete_ball_receipts",
        "dispossessed",
        "miscontrols",
    ]

    features[count_columns] = (
        features[count_columns]
        .fillna(0)
    )

    features["successful_dribble_pct"] = np.where(
        features["dribbles"] > 0,
        (
            features["successful_dribbles"]
            / features["dribbles"]
            * 100
        ),
        np.nan,
    )

    return features