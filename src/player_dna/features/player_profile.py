from __future__ import annotations

import pandas as pd

from player_dna.features.minutes import (
    build_player_minutes,
)

from player_dna.features.passing import (
    build_passing_features,
)

from player_dna.features.positions import (
    build_player_positions,
)

from player_dna.features.possession import (
    build_possession_features,
)

from player_dna.features.defending import (
    build_defending_features,
)

from player_dna.features.attacking import (
    build_attacking_features,
)

from player_dna.features.receiving import (
    build_receiving_features,
)

from player_dna.features.spatial import (
    build_spatial_features,
)


PLAYER_KEYS = [
    "player.id",
    "player.name",
    "team.name",
]


PER_90_COLUMNS = [
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
    "total_carry_distance",
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
    "xg",
    "npxg",
    "xa",

    # Receiving
    "final_third_receipts",
    "box_receipts",
    "receipts_under_pressure",
]


# Raw values that should be zero when
# no corresponding action occurred,
# but that should NOT become per-90 features.
ZERO_ONLY_COLUMNS = [
    # Receiving sanity-check value
    "successful_receipts",

    # Spatial counts / coordinate totals
    "on_ball_actions",
    "on_ball_x_sum",
    "on_ball_y_sum",
    "on_ball_width_sum",

    "defensive_third_actions",
    "middle_third_actions",
    "attacking_third_actions",
    "wide_actions",

    "spatial_receipts",
    "receipt_x_sum",
    "receipt_y_sum",
    "receipt_width_sum",

    "spatial_defensive_actions",
    "defensive_x_sum",
    "defensive_y_sum",
    "defensive_width_sum",
]


def build_player_profile(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one match-level Player DNA profile.

    Combines:
    - minutes
    - positions
    - passing
    - possession
    - defending
    - attacking
    - receiving
    - spatial activity
    """

    minutes = build_player_minutes(
        events
    )

    passing = build_passing_features(
        events
    )

    positions = build_player_positions(
        events
    )

    possession = build_possession_features(
        events
    )

    defending = build_defending_features(
        events
    )

    attacking = build_attacking_features(
        events
    )

    receiving = build_receiving_features(
        events
    )

    spatial = build_spatial_features(
        events
    )

    # --------------------------------------------------
    # Merge feature groups
    # --------------------------------------------------

    profile = (
        minutes
        .merge(
            passing,
            on=PLAYER_KEYS,
            how="left",
        )
        .merge(
            positions,
            on=PLAYER_KEYS,
            how="left",
        )
        .merge(
            possession,
            on=PLAYER_KEYS,
            how="left",
        )
        .merge(
            defending,
            on=PLAYER_KEYS,
            how="left",
        )
        .merge(
            attacking,
            on=PLAYER_KEYS,
            how="left",
        )
        .merge(
            receiving,
            on=PLAYER_KEYS,
            how="left",
        )
        .merge(
            spatial,
            on=PLAYER_KEYS,
            how="left",
        )
    )

    # --------------------------------------------------
    # Missing count / total values
    # --------------------------------------------------

    for column in PER_90_COLUMNS:
        if column not in profile.columns:
            profile[column] = 0.0

    profile[
        PER_90_COLUMNS
    ] = (
        profile[
            PER_90_COLUMNS
        ]
        .fillna(0.0)
    )

    for column in ZERO_ONLY_COLUMNS:
        if column not in profile.columns:
            profile[column] = 0.0

    profile[
        ZERO_ONLY_COLUMNS
    ] = (
        profile[
            ZERO_ONLY_COLUMNS
        ]
        .fillna(0.0)
    )

    # --------------------------------------------------
    # Per-90 features
    # --------------------------------------------------

    per90_features = pd.DataFrame(
        {
            f"{column}_per90": (
                profile[column]
                * 90
                / profile[
                    "minutes_played"
                ]
            )
            for column in PER_90_COLUMNS
        },
        index=profile.index,
    )

    profile = pd.concat(
        [
            profile,
            per90_features,
        ],
        axis=1,
    )

    return profile