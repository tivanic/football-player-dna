from __future__ import annotations

import numpy as np
import pandas as pd

from player_dna.features.positions import (
    add_position_group,
)


PLAYER_KEYS = [
    "player.id",
    "player.name",
    "team.name",
]


COUNT_COLUMNS = [
    # Passing
    "pass_attempts",
    "completed_passes",
    "forward_passes",
    "progressive_passes",
    "final_third_entries",
    "passes_into_box",
    "under_pressure_passes",
    "open_play_passes",
    "crosses",
    "switches",
    "through_balls",
    "shot_assists",
    "goal_assists",

    # Possession
    "carries",
    "progressive_carries",
    "carries_into_final_third",
    "carries_into_box",
    "dribbles",
    "successful_dribbles",
    "ball_receipts",
    "incomplete_ball_receipts",
    "dispossessed",
    "miscontrols",

    # Defending
    "pressures",
    "ball_recoveries",
    "successful_recoveries",
    "offensive_recoveries",
    "tackles",
    "successful_tackles",
    "aerial_duels_lost",
    "interceptions",
    "successful_interceptions",
    "blocks",
    "dribbled_past",
    "fouls_committed",

    # Attacking
    "shots",
    "non_penalty_shots",
    "non_penalty_shots_in_box",
    "goals",
    "non_penalty_goals",

    # Receiving
    "final_third_receipts",
    "box_receipts",
    "receipts_under_pressure",
]


TOTAL_COLUMNS = [
    # Passing
    "total_pass_length",

    # Possession
    "total_carry_distance",

    # Attacking
    "xg",
    "npxg",
    "xa",
]


SPATIAL_COUNT_COLUMNS = [
    "on_ball_actions",

    "defensive_third_actions",
    "middle_third_actions",
    "attacking_third_actions",
    "wide_actions",

    "spatial_receipts",

    "spatial_defensive_actions",
]


SPATIAL_TOTAL_COLUMNS = [
    "on_ball_x_sum",
    "on_ball_y_sum",
    "on_ball_width_sum",

    "receipt_x_sum",
    "receipt_y_sum",
    "receipt_width_sum",

    "defensive_x_sum",
    "defensive_y_sum",
    "defensive_width_sum",
]


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    multiplier: float = 1.0,
) -> pd.Series:
    """
    Divide two Series safely.

    Returns NaN if denominator is zero.
    """

    return pd.Series(
        np.where(
            denominator > 0,
            (
                numerator
                / denominator
                * multiplier
            ),
            np.nan,
        ),
        index=numerator.index,
    )


def aggregate_player_profiles(
    match_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate match-level profiles into
    competition / season Player DNA profiles.

    Raw event totals are summed first.
    Rates and averages are recalculated
    afterwards from the aggregated totals.
    """

    # --------------------------------------------------
    # Primary position across the competition
    # --------------------------------------------------

    position_counts = (
        match_profiles
        .dropna(
            subset=[
                "primary_position",
            ]
        )
        .groupby(
            [
                "player.id",
                "player.name",
                "team.name",
                "primary_position",
            ]
        )
        .size()
        .reset_index(
            name="position_match_count"
        )
    )

    primary_positions = (
        position_counts
        .sort_values(
            "position_match_count",
            ascending=False,
        )
        .drop_duplicates(
            subset=PLAYER_KEYS
        )
        [
            PLAYER_KEYS
            + [
                "primary_position",
            ]
        ]
    )

    # --------------------------------------------------
    # Available raw columns
    # --------------------------------------------------

    available_count_columns = [
        column
        for column in COUNT_COLUMNS
        if column in match_profiles.columns
    ]

    available_total_columns = [
        column
        for column in TOTAL_COLUMNS
        if column in match_profiles.columns
    ]

    available_spatial_counts = [
        column
        for column in SPATIAL_COUNT_COLUMNS
        if column in match_profiles.columns
    ]

    available_spatial_totals = [
        column
        for column in SPATIAL_TOTAL_COLUMNS
        if column in match_profiles.columns
    ]

    # --------------------------------------------------
    # Aggregation
    # --------------------------------------------------

    aggregation = {
        "match_id": "nunique",
        "minutes_played": "sum",
    }

    for column in (
        available_count_columns
        + available_total_columns
        + available_spatial_counts
        + available_spatial_totals
    ):
        aggregation[
            column
        ] = "sum"

    profiles = (
        match_profiles
        .groupby(
            PLAYER_KEYS,
            as_index=False,
        )
        .agg(
            aggregation
        )
        .rename(
            columns={
                "match_id":
                    "matches_played",
            }
        )
    )

    # ==================================================
    # PASSING
    # ==================================================

    if {
        "completed_passes",
        "pass_attempts",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "pass_completion_pct"
        ] = _safe_ratio(
            profiles[
                "completed_passes"
            ],
            profiles[
                "pass_attempts"
            ],
            multiplier=100,
        )

    if {
        "forward_passes",
        "pass_attempts",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "forward_pass_pct"
        ] = _safe_ratio(
            profiles[
                "forward_passes"
            ],
            profiles[
                "pass_attempts"
            ],
            multiplier=100,
        )

    if {
        "total_pass_length",
        "pass_attempts",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "average_pass_length"
        ] = _safe_ratio(
            profiles[
                "total_pass_length"
            ],
            profiles[
                "pass_attempts"
            ],
        )

    # ==================================================
    # POSSESSION
    # ==================================================

    if {
        "total_carry_distance",
        "carries",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "average_carry_length"
        ] = _safe_ratio(
            profiles[
                "total_carry_distance"
            ],
            profiles[
                "carries"
            ],
        )

    if {
        "successful_dribbles",
        "dribbles",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "successful_dribble_pct"
        ] = _safe_ratio(
            profiles[
                "successful_dribbles"
            ],
            profiles[
                "dribbles"
            ],
            multiplier=100,
        )

    if (
        "total_carry_distance"
        in profiles.columns
    ):
        profiles[
            "total_carry_distance_per90"
        ] = (
            profiles[
                "total_carry_distance"
            ]
            * 90
            / profiles[
                "minutes_played"
            ]
        )

    # ==================================================
    # DEFENDING
    # ==================================================

    if {
        "successful_tackles",
        "tackles",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "tackle_success_pct"
        ] = _safe_ratio(
            profiles[
                "successful_tackles"
            ],
            profiles[
                "tackles"
            ],
            multiplier=100,
        )

    if {
        "successful_interceptions",
        "interceptions",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "interception_success_pct"
        ] = _safe_ratio(
            profiles[
                "successful_interceptions"
            ],
            profiles[
                "interceptions"
            ],
            multiplier=100,
        )

    # ==================================================
    # ATTACKING
    # ==================================================

    if {
        "npxg",
        "non_penalty_shots",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "average_npxg_per_shot"
        ] = _safe_ratio(
            profiles[
                "npxg"
            ],
            profiles[
                "non_penalty_shots"
            ],
        )

    # ==================================================
    # SPATIAL
    # ==================================================

    if {
        "on_ball_x_sum",
        "on_ball_actions",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "average_on_ball_x"
        ] = _safe_ratio(
            profiles[
                "on_ball_x_sum"
            ],
            profiles[
                "on_ball_actions"
            ],
        )

        profiles[
            "average_on_ball_y"
        ] = _safe_ratio(
            profiles[
                "on_ball_y_sum"
            ],
            profiles[
                "on_ball_actions"
            ],
        )

        profiles[
            "average_on_ball_width"
        ] = _safe_ratio(
            profiles[
                "on_ball_width_sum"
            ],
            profiles[
                "on_ball_actions"
            ],
        )

        profiles[
            "defensive_third_action_share"
        ] = _safe_ratio(
            profiles[
                "defensive_third_actions"
            ],
            profiles[
                "on_ball_actions"
            ],
            multiplier=100,
        )

        profiles[
            "middle_third_action_share"
        ] = _safe_ratio(
            profiles[
                "middle_third_actions"
            ],
            profiles[
                "on_ball_actions"
            ],
            multiplier=100,
        )

        profiles[
            "attacking_third_action_share"
        ] = _safe_ratio(
            profiles[
                "attacking_third_actions"
            ],
            profiles[
                "on_ball_actions"
            ],
            multiplier=100,
        )

        profiles[
            "wide_action_share"
        ] = _safe_ratio(
            profiles[
                "wide_actions"
            ],
            profiles[
                "on_ball_actions"
            ],
            multiplier=100,
        )

    if {
        "receipt_x_sum",
        "spatial_receipts",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "average_receipt_x"
        ] = _safe_ratio(
            profiles[
                "receipt_x_sum"
            ],
            profiles[
                "spatial_receipts"
            ],
        )

        profiles[
            "average_receipt_y"
        ] = _safe_ratio(
            profiles[
                "receipt_y_sum"
            ],
            profiles[
                "spatial_receipts"
            ],
        )

        profiles[
            "average_receipt_width"
        ] = _safe_ratio(
            profiles[
                "receipt_width_sum"
            ],
            profiles[
                "spatial_receipts"
            ],
        )

    if {
        "defensive_x_sum",
        "spatial_defensive_actions",
    }.issubset(
        profiles.columns
    ):
        profiles[
            "average_defensive_x"
        ] = _safe_ratio(
            profiles[
                "defensive_x_sum"
            ],
            profiles[
                "spatial_defensive_actions"
            ],
        )

        profiles[
            "average_defensive_y"
        ] = _safe_ratio(
            profiles[
                "defensive_y_sum"
            ],
            profiles[
                "spatial_defensive_actions"
            ],
        )

        profiles[
            "average_defensive_width"
        ] = _safe_ratio(
            profiles[
                "defensive_width_sum"
            ],
            profiles[
                "spatial_defensive_actions"
            ],
        )

    # --------------------------------------------------
    # Primary position
    # --------------------------------------------------

    profiles = profiles.merge(
        primary_positions,
        on=PLAYER_KEYS,
        how="left",
    )

    # --------------------------------------------------
    # Per-90 event features
    # --------------------------------------------------

    for column in available_count_columns:
        profiles[
            f"{column}_per90"
        ] = (
            profiles[column]
            * 90
            / profiles[
                "minutes_played"
            ]
        )

    # Continuous totals requiring per-90
    for column in [
        "xg",
        "npxg",
        "xa",
    ]:
        if column in profiles.columns:
            profiles[
                f"{column}_per90"
            ] = (
                profiles[column]
                * 90
                / profiles[
                    "minutes_played"
                ]
            )

    # --------------------------------------------------
    # Position group
    # --------------------------------------------------

    profiles = add_position_group(
        profiles
    )

    return profiles