from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


PASSING_DNA_FEATURES = [
    "pass_attempts_per90",
    "pass_completion_pct",
    "average_pass_length",
    "forward_pass_pct",
    "progressive_passes_per90",
    "final_third_entries_per90",
    "passes_into_box_per90",
    "under_pressure_passes_per90",
    "switches_per90",
    "through_balls_per90",
    "shot_assists_per90",
]


def find_similar_players(
    profiles: pd.DataFrame,
    player_name: str,
    min_minutes: float = 180,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Find players with the most similar passing profiles.

    Players are compared only within the same broad position group.
    Features are standardized before cosine similarity is calculated.
    """

    candidates = profiles[
        profiles["minutes_played"] >= min_minutes
    ].copy()

    target_rows = candidates[
        candidates["player.name"] == player_name
    ]

    if target_rows.empty:
        raise ValueError(
            f"Player '{player_name}' was not found "
            f"with at least {min_minutes} minutes."
        )

    target = target_rows.iloc[0]

    position_group = target["position_group"]

    candidates = candidates[
        candidates["position_group"] == position_group
    ].copy()

    candidates = candidates.reset_index(drop=True)

    features = candidates[
        PASSING_DNA_FEATURES
    ].astype(float)

    # Defensive fallback in case a feature contains missing values.
    features = features.fillna(
        features.median()
    )

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    similarities = cosine_similarity(
        scaled_features
    )

    target_index = candidates.index[
        candidates["player.name"] == player_name
    ][0]

    candidates["similarity"] = (
        similarities[target_index]
    )

    result = (
        candidates[
            candidates["player.name"] != player_name
        ]
        .sort_values(
            "similarity",
            ascending=False,
        )
        .head(top_n)
    )

    return result