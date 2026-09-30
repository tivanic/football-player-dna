from __future__ import annotations

import pandas as pd


RED_CARD_VALUES = {
    "Red Card",
    "Second Yellow",
}


def _event_time_seconds(row: pd.Series) -> float:
    """Convert StatsBomb match minute/second into elapsed seconds."""

    return float(row["minute"] * 60 + row["second"])


def _match_end_seconds(events: pd.DataFrame) -> float:
    """Return the final elapsed match time in seconds."""

    timed_events = events.dropna(subset=["minute", "second"])

    elapsed_seconds = (
        timed_events["minute"] * 60
        + timed_events["second"]
    )

    return float(elapsed_seconds.max())


def build_player_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate minutes played for each player in one match.

    Logic mirrors the general StatsBomb approach:
    - starters begin at 0
    - substitutes begin at their substitution time
    - substituted/red-carded players stop at that event time
    - everyone else stops at match end
    """

    match_end = _match_end_seconds(events)

    players: dict[int, dict] = {}

    # --------------------------------------------------
    # Starting XI
    # --------------------------------------------------

    starting_events = events[
        events["type.name"] == "Starting XI"
    ]

    for _, event in starting_events.iterrows():
        team_id = int(event["team.id"])
        team_name = event["team.name"]

        lineup = event["tactics.lineup"]

        if not isinstance(lineup, list):
            continue

        for item in lineup:
            player = item["player"]

            player_id = int(player["id"])

            players[player_id] = {
                "player.id": player_id,
                "player.name": player["name"],
                "team.id": team_id,
                "team.name": team_name,
                "time_on_seconds": 0.0,
                "time_off_seconds": match_end,
            }

    # --------------------------------------------------
    # Substitutions
    # --------------------------------------------------

    substitutions = events[
        events["type.name"] == "Substitution"
    ]

    for _, event in substitutions.iterrows():
        substitution_time = _event_time_seconds(event)

        outgoing_id = int(event["player.id"])

        replacement_id = int(
            event["substitution.replacement.id"]
        )

        replacement_name = event[
            "substitution.replacement.name"
        ]

        team_id = int(event["team.id"])
        team_name = event["team.name"]

        # Player leaving the pitch
        if outgoing_id in players:
            players[outgoing_id][
                "time_off_seconds"
            ] = substitution_time

        # Player entering the pitch
        players[replacement_id] = {
            "player.id": replacement_id,
            "player.name": replacement_name,
            "team.id": team_id,
            "team.name": team_name,
            "time_on_seconds": substitution_time,
            "time_off_seconds": match_end,
        }

    # --------------------------------------------------
    # Red cards
    # --------------------------------------------------

    for card_column in [
        "bad_behaviour.card.name",
        "foul_committed.card.name",
    ]:
        if card_column not in events.columns:
            continue

        red_cards = events[
            events[card_column].isin(RED_CARD_VALUES)
        ]

        for _, event in red_cards.iterrows():
            if pd.isna(event["player.id"]):
                continue

            player_id = int(event["player.id"])

            if player_id in players:
                players[player_id][
                    "time_off_seconds"
                ] = _event_time_seconds(event)

    # --------------------------------------------------
    # Create dataframe
    # --------------------------------------------------

    minutes = pd.DataFrame(players.values())

    minutes["minutes_played"] = (
        minutes["time_off_seconds"]
        - minutes["time_on_seconds"]
    ) / 60

    minutes["time_on"] = (
        minutes["time_on_seconds"] / 60
    )

    minutes["time_off"] = (
        minutes["time_off_seconds"] / 60
    )

    return minutes[
        [
            "player.id",
            "player.name",
            "team.id",
            "team.name",
            "time_on",
            "time_off",
            "minutes_played",
        ]
    ].sort_values(
        ["team.name", "time_on", "player.name"]
    )