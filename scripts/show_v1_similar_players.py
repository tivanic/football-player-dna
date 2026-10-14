from __future__ import annotations

from pathlib import Path

import pandas as pd

from player_dna.model.player_dna_v1 import (
    MODEL_FEATURES,
    MODEL_NAMES,
    find_similar_players_v1,
)


DATA_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_"
    "player_profiles.parquet"
)


SHOWCASE_PLAYERS = [
    "Mahrez",
    "Özil",
    "Kanté",
    "Kane",
    "Vardy",
    "van Dijk",
]


def main() -> None:
    profiles = pd.read_parquet(
        DATA_PATH
    )

    for player_name in (
        SHOWCASE_PLAYERS
    ):
        print()
        print("=" * 90)

        try:
            (
                target,
                similar,
            ) = find_similar_players_v1(
                profiles,
                player_name=player_name,
                min_minutes=900,
                top_n=5,
            )

        except ValueError as error:
            print(
                f"{player_name}: "
                f"{error}"
            )
            continue

        role = target[
            "position_group"
        ]

        print(
            f"{target['player.name']} "
            f"| {target['team.name']}"
        )

        print(
            f"Position: "
            f"{target['primary_position']}"
        )

        print(
            f"Role: {role}"
        )

        print(
            f"Model: "
            f"{MODEL_NAMES[role]}"
        )

        print(
            f"Features: "
            f"{len(MODEL_FEATURES[role])}"
        )

        print(
            f"Minutes: "
            f"{target['minutes_played']:.0f}"
        )

        print()
        print(
            "TOP 5 SIMILAR PLAYERS"
        )
        print(
            "---------------------"
        )

        print(
            similar[
                [
                    "rank",
                    "player.name",
                    "team.name",
                    "primary_position",
                    "minutes_played",
                    "similarity_score",
                ]
            ]
            .to_string(
                index=False,
                float_format=lambda x:
                f"{x:.3f}",
            )
        )


if __name__ == "__main__":
    main()