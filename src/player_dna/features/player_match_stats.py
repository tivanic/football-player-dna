from __future__ import annotations

import pandas as pd


def build_player_match_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event counts for each player in a single match."""

    #odbaci visak events
    player_events = events.dropna(subset=["player.name"]).copy()
    #grupiraj i broji events
    event_counts = (
        player_events
        .groupby(
            ["player.id", "player.name", "team.name"],
            dropna=False,
        )["type.name"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    wanted_events = [
        "Pass",
        "Carry",
        "Pressure",
        "Ball Recovery",
        "Duel",
        "Dribble",
        "Interception",
        "Shot",
        "Dispossessed",
        "Miscontrol",
    ]

    for event in wanted_events:
        if event not in event_counts.columns:
            event_counts[event] = 0

    columns = [
        "player.id",
        "player.name",
        "team.name",
        *wanted_events,
    ]

    return event_counts[columns]