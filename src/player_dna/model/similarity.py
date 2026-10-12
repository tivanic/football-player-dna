from __future__ import annotations

import numpy as np
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


POSSESSION_DNA_FEATURES = [
    "carries_per90",
    "total_carry_distance_per90",
    "average_carry_length",
    "progressive_carries_per90",
    "carries_into_final_third_per90",
    "carries_into_box_per90",
    "dribbles_per90",
    "successful_dribbles_per90",
    "dispossessed_per90",
    "miscontrols_per90",
]


DEFENDING_DNA_FEATURES = [
    "pressures_per90",
    "ball_recoveries_per90",
    "tackles_per90",
    "successful_tackles_per90",
    "interceptions_per90",
    "blocks_per90",
    "dribbled_past_per90",
    "fouls_committed_per90",
]


PLAYER_DNA_FEATURES = (
    PASSING_DNA_FEATURES
    + POSSESSION_DNA_FEATURES
    + DEFENDING_DNA_FEATURES
)


def find_similar_players(
    profiles: pd.DataFrame,
    player_name: str,
    min_minutes: float = 180,
    top_n: int = 10,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Find players with similar statistical profiles.

    Players are compared only within the same broad position group.
    Selected features are standardized before cosine similarity
    is calculated.
    """

    if feature_columns is None:
        feature_columns = PLAYER_DNA_FEATURES

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

    # IMPORTANT:
    # Use the features passed to this function,
    # not PLAYER_DNA_FEATURES directly.
    features = candidates[
        feature_columns
    ].astype(float)

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

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

    target_matches = candidates.index[
        candidates["player.name"] == player_name
    ]

    if len(target_matches) == 0:
        raise ValueError(
            f"Player '{player_name}' disappeared "
            "after position filtering."
        )

    target_index = target_matches[0]

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