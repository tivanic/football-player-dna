from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from player_dna.model.similarity import PASSING_DNA_FEATURES


DATA_PATH = Path(
    "data/processed/world_cup_2022_player_profiles.parquet"
)

TARGET_PLAYER = "Granit Xhaka"
COMPARE_PLAYER = "Carlos Henrique Casimiro"

MIN_MINUTES = 180


def main() -> None:
    profiles = pd.read_parquet(DATA_PATH)

    target = profiles[
        profiles["player.name"] == TARGET_PLAYER
    ].iloc[0]

    position_group = target["position_group"]

    candidates = profiles[
        (profiles["minutes_played"] >= MIN_MINUTES)
        & (profiles["position_group"] == position_group)
    ].copy()

    features = candidates[
        PASSING_DNA_FEATURES
    ].astype(float)

    features = features.fillna(
        features.median()
    )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(
        scaled,
        columns=PASSING_DNA_FEATURES,
        index=candidates.index,
    )

    target_index = candidates[
        candidates["player.name"] == TARGET_PLAYER
    ].index[0]

    compare_index = candidates[
        candidates["player.name"] == COMPARE_PLAYER
    ].index[0]

    rows = []

    for feature in PASSING_DNA_FEATURES:
        target_raw = candidates.loc[
            target_index,
            feature,
        ]

        compare_raw = candidates.loc[
            compare_index,
            feature,
        ]

        target_z = scaled_df.loc[
            target_index,
            feature,
        ]

        compare_z = scaled_df.loc[
            compare_index,
            feature,
        ]

        rows.append(
            {
                "feature": feature,
                "xhaka": target_raw,
                "comparison": compare_raw,
                "xhaka_z": target_z,
                "comparison_z": compare_z,
                "z_difference": abs(
                    target_z - compare_z
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        "z_difference"
    )

    print(
        f"{TARGET_PLAYER} vs {COMPARE_PLAYER}"
    )
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            formatters={
                "xhaka": lambda value: f"{value:.2f}",
                "comparison": lambda value: f"{value:.2f}",
                "xhaka_z": lambda value: f"{value:+.2f}",
                "comparison_z": lambda value: f"{value:+.2f}",
                "z_difference": lambda value: f"{value:.2f}",
            },
        )
    )


if __name__ == "__main__":
    main()