from __future__ import annotations

import numpy as np
import pandas as pd

from player_dna.data.statsbomb import (
    load_events,
    load_matches,
)


COMPETITION_ID = 2
SEASON_ID = 27
N_MATCHES = 5


ON_BALL_TYPES = {
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
}

DEFENSIVE_TYPES = {
    "Pressure",
    "Ball Recovery",
    "Duel",
    "Interception",
    "Block",
    "Clearance",
}


def coordinate(
    value,
    index: int,
) -> float:
    if (
        isinstance(value, list)
        and len(value) > index
    ):
        return float(
            value[index]
        )

    return np.nan


def main() -> None:
    matches = load_matches(
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
    )

    all_events = []

    for match in (
        matches
        .head(N_MATCHES)
        .itertuples()
    ):
        match_id = int(
            match.match_id
        )

        print(
            f"Loading match "
            f"{match_id}..."
        )

        events = load_events(
            match_id
        )

        events["match_id"] = (
            match_id
        )

        all_events.append(
            events
        )

    events = pd.concat(
        all_events,
        ignore_index=True,
        sort=False,
    )

    # --------------------------------------------------
    # Extract x / y
    # --------------------------------------------------

    events["spatial_x"] = (
        events["location"]
        .apply(
            lambda value:
            coordinate(
                value,
                0,
            )
        )
    )

    events["spatial_y"] = (
        events["location"]
        .apply(
            lambda value:
            coordinate(
                value,
                1,
            )
        )
    )

    events["has_location"] = (
        events["spatial_x"].notna()
        & events["spatial_y"].notna()
    )

    print()
    print(
        f"Total events: "
        f"{len(events)}"
    )

    print()
    print("=" * 80)
    print("LOCATION COVERAGE BY EVENT TYPE")
    print("=" * 80)

    coverage = (
        events
        .groupby(
            "type.name"
        )
        .agg(
            events=(
                "type.name",
                "size",
            ),
            with_location=(
                "has_location",
                "sum",
            ),
        )
    )

    coverage[
        "location_pct"
    ] = (
        coverage[
            "with_location"
        ]
        / coverage[
            "events"
        ]
        * 100
    )

    coverage = (
        coverage
        .sort_values(
            "events",
            ascending=False,
        )
    )

    print(
        coverage
        .to_string()
    )

    # --------------------------------------------------
    # On-ball actions
    # --------------------------------------------------

    on_ball = events[
        events["type.name"].isin(
            ON_BALL_TYPES
        )
        & events["has_location"]
    ].copy()

    print()
    print("=" * 80)
    print("ON-BALL ACTIONS")
    print("=" * 80)

    print(
        on_ball[
            "type.name"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Mean coordinates:"
    )

    print(
        on_ball[
            [
                "spatial_x",
                "spatial_y",
            ]
        ]
        .mean()
        .to_string()
    )

    # --------------------------------------------------
    # Defensive actions
    # --------------------------------------------------

    defensive = events[
        events["type.name"].isin(
            DEFENSIVE_TYPES
        )
        & events["has_location"]
    ].copy()

    print()
    print("=" * 80)
    print("DEFENSIVE ACTIONS")
    print("=" * 80)

    print(
        defensive[
            "type.name"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Mean coordinates:"
    )

    print(
        defensive[
            [
                "spatial_x",
                "spatial_y",
            ]
        ]
        .mean()
        .to_string()
    )

    # --------------------------------------------------
    # Ball receipts
    # --------------------------------------------------

    receipts = events[
        (events["type.name"] == "Ball Receipt*")
        & events["has_location"]
    ].copy()

    if (
        "ball_receipt.outcome.name"
        in receipts.columns
    ):
        receipts = receipts[
            receipts[
                "ball_receipt.outcome.name"
            ].isna()
        ]

    print()
    print("=" * 80)
    print("SUCCESSFUL RECEIPTS")
    print("=" * 80)

    print(
        f"Successful receipts with location: "
        f"{len(receipts)}"
    )

    print()

    print(
        receipts[
            [
                "spatial_x",
                "spatial_y",
            ]
        ]
        .mean()
        .to_string()
    )

    # --------------------------------------------------
    # Per-player sample
    # --------------------------------------------------

    selected_names = [
        "Mesut Özil",
        "Francesc Fàbregas i Soler",
        "Eden Hazard",
        "Diego da Silva Costa",
        "Kurt Happy Zouma",
    ]

    print()
    print("=" * 80)
    print("PLAYER SPATIAL SAMPLE")
    print("=" * 80)

    for player_name in selected_names:

        player_on_ball = on_ball[
            on_ball[
                "player.name"
            ] == player_name
        ]

        player_defensive = defensive[
            defensive[
                "player.name"
            ] == player_name
        ]

        player_receipts = receipts[
            receipts[
                "player.name"
            ] == player_name
        ]

        print()
        print(player_name)

        print(
            f"  On-ball actions: "
            f"{len(player_on_ball)}"
        )

        if not player_on_ball.empty:
            print(
                f"  On-ball avg: "
                f"x="
                f"{player_on_ball['spatial_x'].mean():.1f}, "
                f"y="
                f"{player_on_ball['spatial_y'].mean():.1f}"
            )

        print(
            f"  Successful receipts: "
            f"{len(player_receipts)}"
        )

        if not player_receipts.empty:
            print(
                f"  Receipt avg: "
                f"x="
                f"{player_receipts['spatial_x'].mean():.1f}, "
                f"y="
                f"{player_receipts['spatial_y'].mean():.1f}"
            )

        print(
            f"  Defensive actions: "
            f"{len(player_defensive)}"
        )

        if not player_defensive.empty:
            print(
                f"  Defensive avg: "
                f"x="
                f"{player_defensive['spatial_x'].mean():.1f}, "
                f"y="
                f"{player_defensive['spatial_y'].mean():.1f}"
            )


if __name__ == "__main__":
    main()