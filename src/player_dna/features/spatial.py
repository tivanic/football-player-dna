from __future__ import annotations

import numpy as np
import pandas as pd


PLAYER_KEYS = [
    "player.id",
    "player.name",
    "team.name",
]


ON_BALL_TYPES = {
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
}


DEFENSIVE_TYPES = {
    "Pressure",
    "Ball Recovery",
    "Duel",
    "Interception",
    "Block",
    "Clearance",
}


def _coordinate(
    value,
    index: int,
) -> float:
    """
    Safely extract an x or y coordinate
    from a StatsBomb location value.
    """

    if (
        isinstance(value, list)
        and len(value) > index
    ):
        return float(
            value[index]
        )

    return np.nan


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    multiplier: float = 1.0,
) -> pd.Series:
    """
    Safely divide two Series.

    Returns NaN when the denominator is zero.
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


def _prepare_locations(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add numeric x/y coordinates and
    distance from the horizontal centre line.
    """

    data = events.copy()

    data[
        "spatial_x"
    ] = data[
        "location"
    ].apply(
        lambda value:
        _coordinate(
            value,
            0,
        )
    )

    data[
        "spatial_y"
    ] = data[
        "location"
    ].apply(
        lambda value:
        _coordinate(
            value,
            1,
        )
    )

    data[
        "width_from_center"
    ] = (
        data[
            "spatial_y"
        ]
        - 40.0
    ).abs()

    return data


def build_spatial_features(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Player DNA spatial features.

    Spatial signals are separated into:

    - on-ball location
    - receiving location
    - defensive-action location

    StatsBomb pitch:
        x: 0 -> 120
        y: 0 -> 80

    Pitch thirds:
        defensive: x < 40
        middle:    40 <= x < 80
        attacking: x >= 80

    Wide areas:
        y < 18 or y > 62

    Raw coordinate sums and action counts are
    retained so competition-level averages can
    later be calculated correctly from totals.
    """

    data = _prepare_locations(
        events
    )

    valid_location = (
        data[
            "spatial_x"
        ].notna()
        & data[
            "spatial_y"
        ].notna()
        & data[
            "player.id"
        ].notna()
    )

    # ==================================================
    # ON-BALL SPATIAL PROFILE
    # ==================================================

    on_ball = data[
        valid_location
        & data[
            "type.name"
        ].isin(
            ON_BALL_TYPES
        )
    ].copy()

    if not on_ball.empty:

        on_ball[
            "defensive_third_action"
        ] = (
            on_ball[
                "spatial_x"
            ] < 40
        )

        on_ball[
            "middle_third_action"
        ] = (
            (
                on_ball[
                    "spatial_x"
                ] >= 40
            )
            & (
                on_ball[
                    "spatial_x"
                ] < 80
            )
        )

        on_ball[
            "attacking_third_action"
        ] = (
            on_ball[
                "spatial_x"
            ] >= 80
        )

        on_ball[
            "wide_action"
        ] = (
            (
                on_ball[
                    "spatial_y"
                ] < 18
            )
            | (
                on_ball[
                    "spatial_y"
                ] > 62
            )
        )

        on_ball_features = (
            on_ball
            .groupby(
                PLAYER_KEYS,
                as_index=False,
            )
            .agg(
                on_ball_actions=(
                    "id",
                    "count",
                ),

                on_ball_x_sum=(
                    "spatial_x",
                    "sum",
                ),

                on_ball_y_sum=(
                    "spatial_y",
                    "sum",
                ),

                on_ball_width_sum=(
                    "width_from_center",
                    "sum",
                ),

                defensive_third_actions=(
                    "defensive_third_action",
                    "sum",
                ),

                middle_third_actions=(
                    "middle_third_action",
                    "sum",
                ),

                attacking_third_actions=(
                    "attacking_third_action",
                    "sum",
                ),

                wide_actions=(
                    "wide_action",
                    "sum",
                ),
            )
        )

    else:
        on_ball_features = pd.DataFrame(
            columns=PLAYER_KEYS
        )

    # ==================================================
    # RECEIVING SPATIAL PROFILE
    # ==================================================

    receipts = data[
        valid_location
        & (
            data[
                "type.name"
            ]
            == "Ball Receipt*"
        )
    ].copy()

    # Only successful receipts.
    if (
        "ball_receipt.outcome.name"
        in receipts.columns
    ):
        receipts = receipts[
            receipts[
                "ball_receipt.outcome.name"
            ].isna()
        ].copy()

    if not receipts.empty:

        receipt_features = (
            receipts
            .groupby(
                PLAYER_KEYS,
                as_index=False,
            )
            .agg(
                spatial_receipts=(
                    "id",
                    "count",
                ),

                receipt_x_sum=(
                    "spatial_x",
                    "sum",
                ),

                receipt_y_sum=(
                    "spatial_y",
                    "sum",
                ),

                receipt_width_sum=(
                    "width_from_center",
                    "sum",
                ),
            )
        )

    else:
        receipt_features = pd.DataFrame(
            columns=PLAYER_KEYS
        )

    # ==================================================
    # DEFENSIVE SPATIAL PROFILE
    # ==================================================

    defensive = data[
        valid_location
        & data[
            "type.name"
        ].isin(
            DEFENSIVE_TYPES
        )
    ].copy()

    if not defensive.empty:

        defensive_features = (
            defensive
            .groupby(
                PLAYER_KEYS,
                as_index=False,
            )
            .agg(
                spatial_defensive_actions=(
                    "id",
                    "count",
                ),

                defensive_x_sum=(
                    "spatial_x",
                    "sum",
                ),

                defensive_y_sum=(
                    "spatial_y",
                    "sum",
                ),

                defensive_width_sum=(
                    "width_from_center",
                    "sum",
                ),
            )
        )

    else:
        defensive_features = pd.DataFrame(
            columns=PLAYER_KEYS
        )

    # ==================================================
    # MERGE
    # ==================================================

    spatial = (
        on_ball_features
        .merge(
            receipt_features,
            on=PLAYER_KEYS,
            how="outer",
        )
        .merge(
            defensive_features,
            on=PLAYER_KEYS,
            how="outer",
        )
    )

    # --------------------------------------------------
    # Raw totals
    # --------------------------------------------------

    raw_columns = [
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

    for column in raw_columns:
        if column not in spatial.columns:
            spatial[
                column
            ] = 0.0

        spatial[
            column
        ] = (
            spatial[
                column
            ]
            .fillna(0.0)
        )

    # ==================================================
    # DERIVED SPATIAL FEATURES
    # ==================================================

    spatial[
        "average_on_ball_x"
    ] = _safe_ratio(
        spatial[
            "on_ball_x_sum"
        ],
        spatial[
            "on_ball_actions"
        ],
    )

    spatial[
        "average_on_ball_y"
    ] = _safe_ratio(
        spatial[
            "on_ball_y_sum"
        ],
        spatial[
            "on_ball_actions"
        ],
    )

    spatial[
        "average_on_ball_width"
    ] = _safe_ratio(
        spatial[
            "on_ball_width_sum"
        ],
        spatial[
            "on_ball_actions"
        ],
    )

    spatial[
        "defensive_third_action_share"
    ] = _safe_ratio(
        spatial[
            "defensive_third_actions"
        ],
        spatial[
            "on_ball_actions"
        ],
        multiplier=100,
    )

    spatial[
        "middle_third_action_share"
    ] = _safe_ratio(
        spatial[
            "middle_third_actions"
        ],
        spatial[
            "on_ball_actions"
        ],
        multiplier=100,
    )

    spatial[
        "attacking_third_action_share"
    ] = _safe_ratio(
        spatial[
            "attacking_third_actions"
        ],
        spatial[
            "on_ball_actions"
        ],
        multiplier=100,
    )

    spatial[
        "wide_action_share"
    ] = _safe_ratio(
        spatial[
            "wide_actions"
        ],
        spatial[
            "on_ball_actions"
        ],
        multiplier=100,
    )

    spatial[
        "average_receipt_x"
    ] = _safe_ratio(
        spatial[
            "receipt_x_sum"
        ],
        spatial[
            "spatial_receipts"
        ],
    )

    spatial[
        "average_receipt_y"
    ] = _safe_ratio(
        spatial[
            "receipt_y_sum"
        ],
        spatial[
            "spatial_receipts"
        ],
    )

    spatial[
        "average_receipt_width"
    ] = _safe_ratio(
        spatial[
            "receipt_width_sum"
        ],
        spatial[
            "spatial_receipts"
        ],
    )

    spatial[
        "average_defensive_x"
    ] = _safe_ratio(
        spatial[
            "defensive_x_sum"
        ],
        spatial[
            "spatial_defensive_actions"
        ],
    )

    spatial[
        "average_defensive_y"
    ] = _safe_ratio(
        spatial[
            "defensive_y_sum"
        ],
        spatial[
            "spatial_defensive_actions"
        ],
    )

    spatial[
        "average_defensive_width"
    ] = _safe_ratio(
        spatial[
            "defensive_width_sum"
        ],
        spatial[
            "spatial_defensive_actions"
        ],
    )

    return spatial