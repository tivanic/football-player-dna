from pathlib import Path

import pandas as pd

from player_dna.model.similarity import (
    PASSING_DNA_FEATURES,
    find_similar_players,
)


DATA_PATH = Path(
    "data/processed/world_cup_2022_player_profiles.parquet"
)

PLAYER_NAME = "Granit Xhaka"


def main() -> None:
    profiles = pd.read_parquet(
        DATA_PATH
    )

    target = profiles[
        profiles["player.name"] == PLAYER_NAME
    ].iloc[0]

    print("PLAYER DNA")
    print("----------")
    print(f"Player: {target['player.name']}")
    print(f"Team: {target['team.name']}")
    print(
        f"Position: {target['primary_position']}"
    )
    print(
        f"Position group: "
        f"{target['position_group']}"
    )
    print(
        f"Minutes: "
        f"{target['minutes_played']:.1f}"
    )

    print("\nPASSING FEATURES")
    print("----------------")

    for feature in PASSING_DNA_FEATURES:
        print(
            f"{feature:35s} "
            f"{target[feature]:.2f}"
        )

    similar = find_similar_players(
        profiles=profiles,
        player_name=PLAYER_NAME,
        min_minutes=180,
        top_n=10,
    )

    print("\nMOST SIMILAR PLAYERS")
    print("--------------------")

    columns = [
        "player.name",
        "team.name",
        "primary_position",
        "minutes_played",
        "similarity",
    ]

    print(
        similar[columns]
        .to_string(
            index=False,
            formatters={
                "minutes_played":
                    lambda value: f"{value:.1f}",
                "similarity":
                    lambda value: f"{value:.4f}",
            },
        )
    )


if __name__ == "__main__":
    main()