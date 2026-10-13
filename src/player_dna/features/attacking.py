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


def build_attacking_features(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build shooting, goal-threat and
    expected-assist features from
    StatsBomb event data.
    """

    # ==================================================
    # SHOTS
    # ==================================================

    shots = events[
        events["type.name"] == "Shot"
    ].copy()

    shot_features = pd.DataFrame(
        columns=PLAYER_KEYS
    )

    if not shots.empty:

        shots["shot_x"] = (
            shots["location"]
            .apply(
                lambda value:
                _coordinate(
                    value,
                    0,
                )
            )
        )

        shots["shot_y"] = (
            shots["location"]
            .apply(
                lambda value:
                _coordinate(
                    value,
                    1,
                )
            )
        )

        # StatsBomb pitch:
        # x: 0 -> 120
        # y: 0 -> 80
        #
        # Penalty area approximately:
        # x >= 102
        # 18 <= y <= 62

        shots["in_box"] = (
            (shots["shot_x"] >= 102)
            & (shots["shot_y"] >= 18)
            & (shots["shot_y"] <= 62)
        )

        if (
            "shot.type.name"
            in shots.columns
        ):
            shots[
                "is_penalty"
            ] = (
                shots[
                    "shot.type.name"
                ] == "Penalty"
            )
        else:
            shots[
                "is_penalty"
            ] = False

        if (
            "shot.outcome.name"
            in shots.columns
        ):
            shots[
                "is_goal"
            ] = (
                shots[
                    "shot.outcome.name"
                ] == "Goal"
            )
        else:
            shots[
                "is_goal"
            ] = False

        if (
            "shot.statsbomb_xg"
            in shots.columns
        ):
            shots[
                "shot_xg"
            ] = pd.to_numeric(
                shots[
                    "shot.statsbomb_xg"
                ],
                errors="coerce",
            ).fillna(0.0)
        else:
            shots[
                "shot_xg"
            ] = 0.0

        # Non-penalty metrics are more useful
        # for style analysis because penalty-taking
        # is heavily affected by team role.
        shots[
            "non_penalty_shot"
        ] = (
            ~shots[
                "is_penalty"
            ]
        )

        shots[
            "non_penalty_shot_in_box"
        ] = (
            shots[
                "non_penalty_shot"
            ]
            & shots[
                "in_box"
            ]
        )

        shots[
            "non_penalty_goal"
        ] = (
            shots[
                "is_goal"
            ]
            & shots[
                "non_penalty_shot"
            ]
        )

        shots[
            "npxg_value"
        ] = np.where(
            shots[
                "non_penalty_shot"
            ],
            shots[
                "shot_xg"
            ],
            0.0,
        )

        shot_features = (
            shots
            .groupby(
                PLAYER_KEYS,
                as_index=False,
            )
            .agg(
                shots=(
                    "id",
                    "count",
                ),

                non_penalty_shots=(
                    "non_penalty_shot",
                    "sum",
                ),

                non_penalty_shots_in_box=(
                    "non_penalty_shot_in_box",
                    "sum",
                ),

                goals=(
                    "is_goal",
                    "sum",
                ),

                non_penalty_goals=(
                    "non_penalty_goal",
                    "sum",
                ),

                xg=(
                    "shot_xg",
                    "sum",
                ),

                npxg=(
                    "npxg_value",
                    "sum",
                ),
            )
        )

        shot_features[
            "average_npxg_per_shot"
        ] = np.where(
            shot_features[
                "non_penalty_shots"
            ] > 0,
            (
                shot_features[
                    "npxg"
                ]
                / shot_features[
                    "non_penalty_shots"
                ]
            ),
            np.nan,
        )

    # ==================================================
    # EXPECTED ASSISTS
    # ==================================================

    passes = events[
        events["type.name"] == "Pass"
    ].copy()

    xa_features = pd.DataFrame(
        columns=PLAYER_KEYS
    )

    if (
        not passes.empty
        and "pass.assisted_shot_id"
        in passes.columns
        and not shots.empty
    ):

        assisted_passes = passes[
            passes[
                "pass.assisted_shot_id"
            ].notna()
        ].copy()

        shot_lookup = shots[
            [
                "id",
                "shot_xg",
            ]
        ].rename(
            columns={
                "id": "assisted_shot_id",
                "shot_xg": "assisted_shot_xg",
            }
        )

        assisted_passes = (
            assisted_passes.merge(
                shot_lookup,
                left_on="pass.assisted_shot_id",
                right_on="assisted_shot_id",
                how="left",
            )
        )

        assisted_passes[
            "assisted_shot_xg"
        ] = (
            assisted_passes[
                "assisted_shot_xg"
            ]
            .fillna(0.0)
        )

        xa_features = (
            assisted_passes
            .groupby(
                PLAYER_KEYS,
                as_index=False,
            )
            .agg(
                xa=(
                    "assisted_shot_xg",
                    "sum",
                )
            )
        )

    # ==================================================
    # MERGE SHOOTING + XA
    # ==================================================

    attacking = (
        shot_features.merge(
            xa_features,
            on=PLAYER_KEYS,
            how="outer",
        )
    )

    count_columns = [
        "shots",
        "non_penalty_shots",
        "non_penalty_shots_in_box",
        "goals",
        "non_penalty_goals",
    ]

    total_columns = [
        "xg",
        "npxg",
        "xa",
    ]

    for column in (
        count_columns
        + total_columns
    ):
        if column not in attacking.columns:
            attacking[
                column
            ] = 0.0

        attacking[
            column
        ] = (
            attacking[
                column
            ]
            .fillna(0.0)
        )

    if (
        "average_npxg_per_shot"
        not in attacking.columns
    ):
        attacking[
            "average_npxg_per_shot"
        ] = np.nan

    return attacking