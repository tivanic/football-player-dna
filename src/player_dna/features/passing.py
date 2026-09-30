from __future__ import annotations

import numpy as np
import pandas as pd


SET_PIECE_PASS_TYPES = {
    "Throw-in",
    "Goal Kick",
    "Free Kick",
    "Corner",
    "Kick Off",
}


def _coordinate(value: object, index: int) -> float:
    """Safely extract x or y from a StatsBomb location."""

    if isinstance(value, list) and len(value) > index:
        return float(value[index])

    return np.nan


def _boolean_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a boolean column, treating missing values as False."""

    if column not in df.columns:
        return pd.Series(False, index=df.index)

    return df[column].fillna(False).astype(bool)


def build_passing_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build match-level passing features for every player."""

    passes = events[
        events["type.name"] == "Pass"
    ].dropna(
        subset=["player.id", "player.name", "team.name"]
    ).copy()

    # --------------------------------------------------
    # Coordinates
    # --------------------------------------------------

    passes["start_x"] = passes["location"].apply(
        lambda value: _coordinate(value, 0)
    )
    passes["start_y"] = passes["location"].apply(
        lambda value: _coordinate(value, 1)
    )

    passes["end_x"] = passes["pass.end_location"].apply(
        lambda value: _coordinate(value, 0)
    )
    passes["end_y"] = passes["pass.end_location"].apply(
        lambda value: _coordinate(value, 1)
    )

    # --------------------------------------------------
    # Basic pass properties
    # --------------------------------------------------

    passes["completed"] = passes["pass.outcome.name"].isna()

    passes["x_progression"] = (
        passes["end_x"] - passes["start_x"]
    )

    passes["forward"] = (
        passes["x_progression"] > 0
    )

    # Project-specific baseline definition:
    # completed pass moving the ball at least 15 StatsBomb
    # pitch units toward the opponent goal.
    passes["progressive"] = (
        passes["completed"]
        & (passes["x_progression"] >= 15)
    )

    # --------------------------------------------------
    # Pitch zones
    # --------------------------------------------------

    passes["final_third_entry"] = (
        passes["completed"]
        & (passes["start_x"] < 80)
        & (passes["end_x"] >= 80)
    )

    start_in_box = (
        (passes["start_x"] >= 102)
        & passes["start_y"].between(18, 62)
    )

    end_in_box = (
        (passes["end_x"] >= 102)
        & passes["end_y"].between(18, 62)
    )

    passes["into_box"] = (
        passes["completed"]
        & ~start_in_box
        & end_in_box
    )

    # --------------------------------------------------
    # Context
    # --------------------------------------------------

    passes["under_pressure_pass"] = _boolean_column(
        passes,
        "under_pressure",
    )

    passes["cross"] = _boolean_column(
        passes,
        "pass.cross",
    )

    passes["switch"] = _boolean_column(
        passes,
        "pass.switch",
    )

    passes["through_ball"] = _boolean_column(
        passes,
        "pass.through_ball",
    )

    passes["shot_assist"] = _boolean_column(
        passes,
        "pass.shot_assist",
    )

    passes["goal_assist"] = _boolean_column(
        passes,
        "pass.goal_assist",
    )

    passes["open_play"] = ~passes[
        "pass.type.name"
    ].isin(SET_PIECE_PASS_TYPES)

    # --------------------------------------------------
    # Aggregate by player
    # --------------------------------------------------

    group_columns = [
        "player.id",
        "player.name",
        "team.name",
    ]

    features = (
        passes
        .groupby(group_columns)
        .agg(
            pass_attempts=("id", "size"),
            completed_passes=("completed", "sum"),
            average_pass_length=("pass.length", "mean"),
            forward_passes=("forward", "sum"),
            progressive_passes=("progressive", "sum"),
            final_third_entries=("final_third_entry", "sum"),
            passes_into_box=("into_box", "sum"),
            under_pressure_passes=("under_pressure_pass", "sum"),
            open_play_passes=("open_play", "sum"),
            crosses=("cross", "sum"),
            switches=("switch", "sum"),
            through_balls=("through_ball", "sum"),
            shot_assists=("shot_assist", "sum"),
            goal_assists=("goal_assist", "sum"),
        )
        .reset_index()
    )

    features["pass_completion_pct"] = (
        features["completed_passes"]
        / features["pass_attempts"]
        * 100
    )

    features["forward_pass_pct"] = (
        features["forward_passes"]
        / features["pass_attempts"]
        * 100
    )

    return features