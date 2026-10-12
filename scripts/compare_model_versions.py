from pathlib import Path

import pandas as pd

from player_dna.model.similarity import (
    PASSING_DNA_FEATURES,
    POSSESSION_DNA_FEATURES,
    DEFENDING_DNA_FEATURES,
    find_similar_players,
)


DATA_PATH = Path(
    "data/processed/world_cup_2022_player_profiles.parquet"
)

PLAYER_NAME = "Granit Xhaka"
MIN_MINUTES = 180
TOP_N = 10


MODEL_VERSIONS = {
    "v0.1 Passing": (
        PASSING_DNA_FEATURES
    ),
    "v0.2 Passing + Possession": (
        PASSING_DNA_FEATURES
        + POSSESSION_DNA_FEATURES
    ),
    "v0.3 Passing + Possession + Defending": (
        PASSING_DNA_FEATURES
        + POSSESSION_DNA_FEATURES
        + DEFENDING_DNA_FEATURES
    ),
}


def main() -> None:
    profiles = pd.read_parquet(DATA_PATH)

    for version_name, features in MODEL_VERSIONS.items():
        print()
        print("=" * 80)
        print(version_name)
        print(f"Features: {len(features)}")
        print("=" * 80)

        print("Feature set:")
        print(features)
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