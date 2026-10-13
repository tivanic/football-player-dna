from __future__ import annotations

import numpy as np
import pandas as pd


PLAYER_KEYS = [
    "player.id",
    "player.name",
    "team.name",
]


def _coordinate(
    value,
    index: int,
) -> float:
    """
    Safely extract one coordinate
    from a StatsBomb location list.
    """

    if (
        isinstance(value, list)
        and len(value) > index
    ):
        return float(
            value[index]
        )

    return np.nan


def build_receiving_features(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build receiving-location features from
    StatsBomb Ball Receipt events.

    Zone-based features count successful
    receptions only.

    StatsBomb pitch:
    x: 0 -> 120
    y: 0 -> 80

    Final third:
    x >= 80

    Penalty area:
    x >= 102
    18 <= y <= 62
    """

    receipts = events[
        events["type.name"]
        == "Ball Receipt*"
    ].copy()

    if receipts.empty:
        return pd.DataFrame(
            columns=PLAYER_KEYS
            + [
                "successful_receipts",
                "final_third_receipts",
                "box_receipts",
                "receipts_under_pressure",
            ]
        )

    # --------------------------------------------------
    # Receipt outcome
    # --------------------------------------------------

    if (
        "ball_receipt.outcome.name"
        in receipts.columns
    ):
        receipts[
            "successful_receipt"
        ] = (
            receipts[
                "ball_receipt.outcome.name"
            ].isna()
        )
    else:
        receipts[
            "successful_receipt"
        ] = True

    # --------------------------------------------------
    # Location
    # --------------------------------------------------

    receipts[
        "receipt_x"
    ] = receipts[
        "location"
    ].apply(
        lambda value:
        _coordinate(
            value,
            0,
        )
    )

    receipts[
        "receipt_y"
    ] = receipts[
        "location"
    ].apply(
        lambda value:
        _coordinate(
            value,
            1,
        )
    )

    # --------------------------------------------------
    # Receiving zones
    # --------------------------------------------------

    receipts[
        "final_third_receipt"
    ] = (
        receipts[
            "successful_receipt"
        ]
        & (
            receipts[
                "receipt_x"
            ] >= 80
        )
    )

    receipts[
        "box_receipt"
    ] = (
        receipts[
            "successful_receipt"
        ]
        & (
            receipts[
                "receipt_x"
            ] >= 102
        )
        & (
            receipts[
                "receipt_y"
            ] >= 18
        )
        & (
            receipts[
                "receipt_y"
            ] <= 62
        )
    )

    # --------------------------------------------------
    # Receiving under pressure
    # --------------------------------------------------

    if (
        "under_pressure"
        in receipts.columns
    ):
        under_pressure = (
            receipts[
                "under_pressure"
            ]
            .fillna(False)
            .astype(bool)
        )
    else:
        under_pressure = pd.Series(
            False,
            index=receipts.index,
        )

    # Only successful receptions count here.
    receipts[
        "receipt_under_pressure"
    ] = (
        receipts[
            "successful_receipt"
        ]
        & under_pressure
    )

    # --------------------------------------------------
    # Aggregate by player
    # --------------------------------------------------

    receiving = (
        receipts
        .groupby(
            PLAYER_KEYS,
            as_index=False,
        )
        .agg(
            successful_receipts=(
                "successful_receipt",
                "sum",
            ),

            final_third_receipts=(
                "final_third_receipt",
                "sum",
            ),

            box_receipts=(
                "box_receipt",
                "sum",
            ),

            receipts_under_pressure=(
                "receipt_under_pressure",
                "sum",
            ),
        )
    )

    return receiving