from __future__ import annotations

import pandas as pd

POSITION_GROUPS = {
    "Goalkeeper": "GK",

    "Right Center Back": "CB",
    "Center Back": "CB",
    "Left Center Back": "CB",

    "Right Back": "FB",
    "Left Back": "FB",
    "Right Wing Back": "FB",
    "Left Wing Back": "FB",

    "Right Defensive Midfield": "DM_CM",
    "Left Defensive Midfield": "DM_CM",
    "Center Defensive Midfield": "DM_CM",
    "Right Center Midfield": "DM_CM",
    "Center Midfield": "DM_CM",
    "Left Center Midfield": "DM_CM",

    "Right Attacking Midfield": "AM_W",
    "Center Attacking Midfield": "AM_W",
    "Left Attacking Midfield": "AM_W",
    "Right Wing": "AM_W",
    "Left Wing": "AM_W",

    "Right Forward": "FW",
    "Center Forward": "FW",
    "Left Forward": "FW",
    "Secondary Striker": "FW",
}

def build_player_positions(events: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the positions in which each player appears during a match.

    The primary position is the most frequently occurring event position.
    """

    player_events = events.dropna(
        subset=[
            "player.id",
            "player.name",
            "team.name",
            "position.name",
        ]
    ).copy()

    position_counts = (
        player_events
        .groupby(
            [
                "player.id",
                "player.name",
                "team.name",
                "position.name",
            ]
        )
        .size()
        .reset_index(name="position_event_count")
    )

    primary_positions = (
        position_counts
        .sort_values(
            "position_event_count",
            ascending=False,
        )
        .drop_duplicates(
            subset=["player.id", "team.name"]
        )
        .rename(
            columns={
                "position.name": "primary_position",
            }
        )
    )

    return primary_positions[
        [
            "player.id",
            "player.name",
            "team.name",
            "primary_position",
        ]
    ]

def add_position_group(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Map detailed StatsBomb positions to broader role groups."""

    profiles = profiles.copy()

    profiles["position_group"] = (
        profiles["primary_position"]
        .map(POSITION_GROUPS)
        .fillna("OTHER")
    )

    return profiles