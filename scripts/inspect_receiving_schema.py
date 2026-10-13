from __future__ import annotations

import pandas as pd

from player_dna.data.statsbomb import (
    load_events,
    load_matches,
)


COMPETITION_ID = 2
SEASON_ID = 27
N_MATCHES = 5


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

        events[
            "match_id"
        ] = match_id

        all_events.append(
            events
        )

    events = pd.concat(
        all_events,
        ignore_index=True,
        sort=False,
    )

    receipts = events[
        events["type.name"]
        == "Ball Receipt*"
    ].copy()

    print()
    print(
        f"Ball Receipt events: "
        f"{len(receipts)}"
    )

    print()
    print("=" * 80)
    print("RECEIPT-RELATED COLUMNS")
    print("=" * 80)

    relevant_columns = [
        column
        for column in receipts.columns
        if (
            column.startswith(
                "ball_receipt."
            )
            or column
            in {
                "location",
                "under_pressure",
                "player.id",
                "player.name",
                "team.name",
            }
        )
    ]

    for column in relevant_columns:
        print(column)

    sample_columns = [
        column
        for column in [
            "match_id",
            "minute",
            "second",
            "player.id",
            "player.name",
            "team.name",
            "location",
            "under_pressure",
            "ball_receipt.outcome.name",
        ]
        if column in receipts.columns
    ]

    print()
    print("=" * 80)
    print("RECEIPT SAMPLE")
    print("=" * 80)

    print(
        receipts[
            sample_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    if (
        "ball_receipt.outcome.name"
        in receipts.columns
    ):
        print()
        print("RECEIPT OUTCOMES")
        print("----------------")

        print(
            receipts[
                "ball_receipt.outcome.name"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    if (
        "under_pressure"
        in receipts.columns
    ):
        print()
        print("UNDER PRESSURE")
        print("--------------")

        print(
            receipts[
                "under_pressure"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


if __name__ == "__main__":
    main()