from pathlib import Path

import numpy as np
import pandas as pd

from player_dna.model.similarity import (
    PLAYER_DNA_FEATURES,
)


DATA_PATH = Path(
    "data/processed/world_cup_2022_player_profiles.parquet"
)

POSITION_GROUP = "DM_CM"
MIN_MINUTES = 180
CORRELATION_THRESHOLD = 0.80


def main() -> None:
    profiles = pd.read_parquet(DATA_PATH)

    players = profiles[
        (profiles["position_group"] == POSITION_GROUP)
        & (profiles["minutes_played"] >= MIN_MINUTES)
    ].copy()

    features = players[
        PLAYER_DNA_FEATURES
    ].astype(float)

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print(
        f"Players analysed: {len(players)}"
    )

    print(
        f"Features analysed: {len(PLAYER_DNA_FEATURES)}"
    )

    correlations = features.corr()

    pairs = []

    for i, feature_a in enumerate(
        PLAYER_DNA_FEATURES
    ):
        for feature_b in PLAYER_DNA_FEATURES[
            i + 1:
        ]:
            correlation = correlations.loc[
                feature_a,
                feature_b,
            ]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= CORRELATION_THRESHOLD:
                pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "correlation": correlation,
                        "absolute_correlation":
                            abs(correlation),
                    }
                )

    result = pd.DataFrame(pairs)

    if result.empty:
        print(
            "\nNo feature pairs above "
            f"{CORRELATION_THRESHOLD:.2f}"
        )
        return

    result = result.sort_values(
        "absolute_correlation",
        ascending=False,
    )

    print(
        "\nHIGHLY CORRELATED FEATURES"
    )

    print(
        "--------------------------"
    )

    print(
        result[
            [
                "feature_a",
                "feature_b",
                "correlation",
            ]
        ].to_string(
            index=False,
            formatters={
                "correlation":
                    lambda value: f"{value:+.3f}",
            },
        )
    )


if __name__ == "__main__":
    main()