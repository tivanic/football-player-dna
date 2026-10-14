from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import (
    cosine_similarity,
)
from sklearn.preprocessing import (
    StandardScaler,
)


# ======================================================
# SHARED FEATURE SETS
# ======================================================

FULL_FEATURES = [
    # Passing
    "pass_attempts_per90",
    "pass_completion_pct",
    "average_pass_length",
    "forward_pass_pct",
    "progressive_passes_per90",
    "final_third_entries_per90",
    "passes_into_box_per90",
    "under_pressure_passes_per90",
    "crosses_per90",
    "switches_per90",
    "through_balls_per90",
    "shot_assists_per90",

    # Possession
    "total_carry_distance_per90",
    "average_carry_length",
    "progressive_carries_per90",
    "carries_into_final_third_per90",
    "carries_into_box_per90",
    "dribbles_per90",
    "successful_dribbles_per90",
    "ball_receipts_per90",
    "incomplete_ball_receipts_per90",
    "dispossessed_per90",
    "miscontrols_per90",

    # Defending
    "pressures_per90",
    "ball_recoveries_per90",
    "successful_recoveries_per90",
    "offensive_recoveries_per90",
    "tackles_per90",
    "successful_tackles_per90",
    "aerial_duels_lost_per90",
    "interceptions_per90",
    "successful_interceptions_per90",
    "blocks_per90",
    "dribbled_past_per90",
    "fouls_committed_per90",

    # Attacking
    "non_penalty_shots_per90",
    "non_penalty_shots_in_box_per90",
    "npxg_per90",
    "average_npxg_per_shot",

    # Creation
    "xa_per90",

    # Receiving
    "final_third_receipts_per90",
    "box_receipts_per90",
    "receipts_under_pressure_per90",

    # Spatial
    "average_on_ball_x",
    "average_on_ball_width",
    "average_receipt_x",
    "average_receipt_width",
    "average_defensive_x",
    "average_defensive_width",
    "attacking_third_action_share",
    "wide_action_share",
]


REDUCED_FEATURES = [
    # Distribution / progression
    "pass_attempts_per90",
    "pass_completion_pct",
    "average_pass_length",
    "forward_pass_pct",
    "progressive_passes_per90",
    "passes_into_box_per90",
    "under_pressure_passes_per90",
    "crosses_per90",
    "switches_per90",
    "through_balls_per90",

    # Creation
    "xa_per90",

    # Carrying
    "average_carry_length",
    "progressive_carries_per90",
    "carries_into_box_per90",
    "dribbles_per90",
    "dispossessed_per90",
    "miscontrols_per90",

    # Defending
    "pressures_per90",
    "ball_recoveries_per90",
    "tackles_per90",
    "interceptions_per90",
    "blocks_per90",
    "dribbled_past_per90",
    "fouls_committed_per90",

    # Attacking
    "non_penalty_shots_per90",
    "npxg_per90",
    "average_npxg_per_shot",

    # Receiving
    "final_third_receipts_per90",
    "box_receipts_per90",
    "receipts_under_pressure_per90",

    # Spatial
    "average_on_ball_x",
    "average_on_ball_width",
    "average_defensive_x",
    "average_defensive_width",
]


CB_FEATURES = [
    "pass_attempts_per90",
    "pass_completion_pct",
    "average_pass_length",
    "forward_pass_pct",
    "progressive_passes_per90",
    "under_pressure_passes_per90",
    "switches_per90",

    "average_carry_length",
    "progressive_carries_per90",

    "pressures_per90",
    "ball_recoveries_per90",
    "tackles_per90",
    "interceptions_per90",
    "blocks_per90",
    "dribbled_past_per90",
    "fouls_committed_per90",

    "average_on_ball_x",
    "average_on_ball_width",
    "average_defensive_x",
    "average_defensive_width",
]


FB_FEATURES = [
    "pass_attempts_per90",
    "pass_completion_pct",
    "average_pass_length",
    "forward_pass_pct",
    "progressive_passes_per90",
    "passes_into_box_per90",
    "under_pressure_passes_per90",
    "crosses_per90",
    "switches_per90",
    "xa_per90",

    "average_carry_length",
    "progressive_carries_per90",
    "carries_into_box_per90",
    "dribbles_per90",
    "dispossessed_per90",
    "miscontrols_per90",

    "pressures_per90",
    "ball_recoveries_per90",
    "tackles_per90",
    "interceptions_per90",
    "blocks_per90",
    "dribbled_past_per90",
    "fouls_committed_per90",

    "final_third_receipts_per90",
    "box_receipts_per90",
    "receipts_under_pressure_per90",

    "average_on_ball_x",
    "average_on_ball_width",
    "average_defensive_x",
    "average_defensive_width",
]


MODEL_FEATURES = {
    "CB": CB_FEATURES,
    "FB": FB_FEATURES,

    # Full model had a meaningful advantage
    # in paired DM/CM evaluation.
    "DM_CM": FULL_FEATURES,

    # Reduced models retained essentially
    # the same performance with less
    # redundant weighting.
    "AM_W": REDUCED_FEATURES,
    "FW": REDUCED_FEATURES,
}


MODEL_NAMES = {
    "CB": "ROLE_AWARE_20",
    "FB": "ROLE_AWARE_30",
    "DM_CM": "FULL_51",
    "AM_W": "REDUCED_34",
    "FW": "REDUCED_34",
}


def _find_target(
    profiles: pd.DataFrame,
    player_name: str,
    team_name: str | None = None,
) -> pd.DataFrame:
    """
    Find a player using an exact
    case-insensitive name first.

    If there is no exact match,
    try a substring search.
    """

    names = (
        profiles[
            "player.name"
        ]
        .astype(str)
    )

    exact = profiles[
        names.str.casefold()
        == player_name.casefold()
    ].copy()

    if exact.empty:
        exact = profiles[
            names.str.contains(
                player_name,
                case=False,
                regex=False,
                na=False,
            )
        ].copy()

    if team_name is not None:
        exact = exact[
            exact[
                "team.name"
            ]
            .astype(str)
            .str.casefold()
            == team_name.casefold()
        ]

    if exact.empty:
        raise ValueError(
            f"Player not found: "
            f"{player_name}"
        )

    if len(exact) > 1:
        matches = (
            exact[
                [
                    "player.name",
                    "team.name",
                ]
            ]
            .to_string(
                index=False
            )
        )

        raise ValueError(
            "Player search is ambiguous:\n"
            f"{matches}"
        )

    return exact


def find_similar_players_v1(
    profiles: pd.DataFrame,
    player_name: str,
    team_name: str | None = None,
    min_minutes: float = 900,
    top_n: int = 5,
) -> tuple[
    pd.Series,
    pd.DataFrame,
]:
    """
    Find Player DNA v1 nearest neighbours.

    Similarity is calculated only against
    players in the same broad position group.

    The cosine score is a model similarity
    value, NOT a percentage probability.
    """

    eligible = profiles[
        profiles[
            "minutes_played"
        ] >= min_minutes
    ].copy()

    target_frame = _find_target(
        eligible,
        player_name,
        team_name,
    )

    target = (
        target_frame
        .iloc[0]
    )

    role = target[
        "position_group"
    ]

    if role not in MODEL_FEATURES:
        raise ValueError(
            f"No Player DNA v1 model "
            f"for role: {role}"
        )

    features = MODEL_FEATURES[
        role
    ]

    missing_features = [
        feature
        for feature in features
        if feature not in eligible.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )

    candidates = eligible[
        eligible[
            "position_group"
        ] == role
    ].copy()

    candidates = (
        candidates
        .reset_index(
            drop=True
        )
    )

    values = (
        candidates[
            features
        ]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    medians = (
        values
        .median()
        .fillna(0.0)
    )

    values = values.fillna(
        medians
    )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        values
    )

    target_mask = (
        (
            candidates[
                "player.id"
            ]
            == target[
                "player.id"
            ]
        )
        & (
            candidates[
                "team.name"
            ]
            == target[
                "team.name"
            ]
        )
    )

    target_indices = np.flatnonzero(
        target_mask.to_numpy()
    )

    if len(target_indices) != 1:
        raise ValueError(
            "Could not uniquely locate "
            "target in candidate population."
        )

    target_index = int(
        target_indices[0]
    )

    similarities = (
        cosine_similarity(
            scaled[
                target_index:
                target_index + 1
            ],
            scaled,
        )[0]
    )

    results = candidates[
        [
            "player.id",
            "player.name",
            "team.name",
            "primary_position",
            "position_group",
            "minutes_played",
        ]
    ].copy()

    results[
        "similarity_score"
    ] = similarities

    # Exclude the player from their own results.
    results = results[
        ~target_mask
    ].copy()

    results = (
        results
        .sort_values(
            "similarity_score",
            ascending=False,
        )
        .head(
            top_n
        )
        .reset_index(
            drop=True
        )
    )

    results.insert(
        0,
        "rank",
        np.arange(
            1,
            len(results) + 1,
        ),
    )

    return (
        target,
        results,
    )