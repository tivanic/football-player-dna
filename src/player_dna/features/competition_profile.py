from __future__ import annotations

import pandas as pd
from player_dna.features.positions import add_position_group

COUNT_COLUMNS = [
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


def aggregate_player_profiles(
    match_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate match-level player profiles into one profile
    per player/team across many matches.
    """
    #
    position_counts = (
        match_profiles
        .dropna(subset=["primary_position"])
        .groupby(
            [
                "player.id",
                "player.name",
                "team.name",
                "primary_position",
            ]
        )
        .size()
        .reset_index(name="position_match_count")
    )

    primary_positions = (
        position_counts
        .sort_values(
            "position_match_count",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "player.id",
                "player.name",
                "team.name",
            ]
        )
        [
            [
                "player.id",
                "player.name",
                "team.name",
                "primary_position",
            ]
        ]
    )
    #

    aggregation = {
        "match_id": "nunique",
        "minutes_played": "sum",
        "total_pass_length": "sum",
    }

    for column in COUNT_COLUMNS:
        aggregation[column] = "sum"

    profiles = (
        match_profiles
        .groupby(
            [
                "player.id",
                "player.name",
                "team.name",
            ],
            as_index=False,
        )
        .agg(aggregation)
        .rename(
            columns={
                "match_id": "matches_played",
            }
        )
    )

    # -----------------------------------------------
    # Rates / averages calculated from TOTALS
    # -----------------------------------------------

    profiles["pass_completion_pct"] = (
        profiles["completed_passes"]
        / profiles["pass_attempts"]
        * 100
    )

    profiles["forward_pass_pct"] = (
        profiles["forward_passes"]
        / profiles["pass_attempts"]
        * 100
    )

    profiles["average_pass_length"] = (
        profiles["total_pass_length"]
        / profiles["pass_attempts"]
    )
    #
    profiles = profiles.merge(
    primary_positions,
    on=[
        "player.id",
        "player.name",
        "team.name",
    ],
    how="left",
)
    #

    # -----------------------------------------------
    # Per-90 features
    # -----------------------------------------------

    for column in COUNT_COLUMNS:
        profiles[f"{column}_per90"] = (
            profiles[column]
            * 90
            / profiles["minutes_played"]
        )

    profiles = add_position_group(profiles)
    return profiles