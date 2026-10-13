from __future__ import annotations

from player_dna.data.statsbomb import (
    load_events,
)

from player_dna.features.receiving import (
    build_receiving_features,
)


MATCH_ID = 3754217


def main() -> None:
    events = load_events(
        MATCH_ID
    )

    receiving = (
        build_receiving_features(
            events
        )
    )

    print(
        "RECEIVING FEATURES"
    )
    print(
        "------------------"
    )

    columns = [
        "player.name",
        "team.name",
        "successful_receipts",
        "final_third_receipts",
        "box_receipts",
        "receipts_under_pressure",
    ]

    print(
        receiving[
            columns
        ]
        .sort_values(
            [
                "final_third_receipts",
                "box_receipts",
                "receipts_under_pressure",
            ],
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()