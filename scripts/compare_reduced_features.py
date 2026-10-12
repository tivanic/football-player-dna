from pathlib import Path

import pandas as pd

from player_dna.model.similarity import (
    PLAYER_DNA_FEATURES,
    REDUCED_PLAYER_DNA_FEATURES,
    find_similar_players,
)


DATA_PATH = Path(
    "data/processed/world_cup_2022_player_profiles.parquet"
)

PLAYER_NAME = "Granit Xhaka"
MIN_MINUTES = 180
TOP_N = 10


FEATURE_SETS = {
    "v0.3 Full": PLAYER_DNA_FEATURES,
    "v0.3 Reduced": REDUCED_PLAYER_DNA_FEATURES,
}


def main() -> None:
    profiles = pd.read_parquet(DATA_PATH)

    for name, features in FEATURE_SETS.items():
        print()
        print("=" * 80)
        print(name)
        print(f"Features: {len(features)}")
        print("=" * 80)

        similar = find_similar_players(
            profiles=profiles,
            player_name=PLAYER_NAME,
            min_minutes=MIN_MINUTES,
            top_n=TOP_N,
            feature_columns=features,
        )

        result = similar[
            [
                "player.name",
                "team.name",
                "primary_position",
                "similarity",
            ]
        ].copy()

        result.insert(
            0,
            "rank",
            range(1, len(result) + 1),
        )

        print(
            result.to_string(
                index=False,
                formatters={
                    "similarity":
                        lambda value: f"{value:.4f}",
                },
            )
        )


if __name__ == "__main__":
    main()