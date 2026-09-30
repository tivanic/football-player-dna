from __future__ import annotations

import pandas as pd

from player_dna.features.minutes import build_player_minutes
from player_dna.features.passing import build_passing_features


PER_90_COLUMNS = [
    "pass_attempts",
    "completed_passes",
    "forward_passes",
    "progressive_passes",
    "final_third_entries",
    "passes_into_box",
    "under_pressure_passes",
    "open_play_passes",
    "crosses",
    "switches",
    "through_balls",
    "shot_assists",
    "goal_assists",
]


def build_player_profile(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build a match-level player profile by combining minutes played
    with passing features and creating per-90 metrics.
    """

    minutes = build_player_minutes(events)
    passing = build_passing_features(events)

    profile = minutes.merge(
        passing,
        on=[
            "player.id",
            "player.name",
            "team.name",
        ],
        how="left",
    )

    # Players who did not attempt a pass will have missing
    # passing values after the merge.
    numeric_passing_columns = [
        column
        for column in passing.columns
        if column not in {
            "player.id",
            "player.name",
            "team.name",
        }
    ]

    profile[numeric_passing_columns] = (
        profile[numeric_passing_columns]
        .fillna(0)
    )

    for column in PER_90_COLUMNS:
        profile[f"{column}_per90"] = (
            profile[column]
            * 90
            / profile["minutes_played"]
        )

    return profile