from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import (
    cosine_similarity,
)
from sklearn.preprocessing import (
    StandardScaler,
)

from player_dna.features.competition_profile import (
    aggregate_player_profiles,
)


MATCH_PROFILES_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_"
    "match_profiles.parquet"
)

SEASON_PROFILES_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_"
    "player_profiles.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_"
    "role_model_evaluation.csv"
)


PLAYER_KEYS = [
    "player.id",
    "player.name",
    "team.name",
]


ROLE_GROUPS = [
    "CB",
    "FB",
    "DM_CM",
    "AM_W",
    "FW",
]


N_REPEATS = 50
BASE_SEED = 42

MIN_SEASON_MINUTES = 1500
MIN_MATCHES = 12
MIN_SPLIT_MINUTES = 600


# ======================================================
# FULL 51-FEATURE MODEL
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


# ======================================================
# OLD 27-FEATURE BASELINE
# ======================================================

LEGACY_FEATURES = [
    # Passing
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

    # Possession
    "total_carry_distance_per90",
    "average_carry_length",
    "progressive_carries_per90",
    "carries_into_final_third_per90",
    "carries_into_box_per90",
    "dribbles_per90",
    "successful_dribbles_per90",
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
]


# ======================================================
# SHARED REDUCED MODEL
# ======================================================

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


# ======================================================
# ROLE-AWARE MODEL
#
# These are hypotheses to TEST.
# They are not yet the final Player DNA feature sets.
# ======================================================

ROLE_FEATURES = {
    "CB": [
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
    ],

    "FB": [
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
    ],

    "DM_CM": [
        "pass_attempts_per90",
        "pass_completion_pct",
        "average_pass_length",
        "forward_pass_pct",
        "progressive_passes_per90",
        "passes_into_box_per90",
        "under_pressure_passes_per90",
        "switches_per90",
        "through_balls_per90",
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
        "receipts_under_pressure_per90",

        "average_on_ball_x",
        "average_on_ball_width",
        "average_defensive_x",
        "average_defensive_width",
    ],

    "AM_W": [
        "pass_attempts_per90",
        "pass_completion_pct",
        "average_pass_length",
        "progressive_passes_per90",
        "passes_into_box_per90",
        "under_pressure_passes_per90",
        "crosses_per90",
        "through_balls_per90",
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

        "non_penalty_shots_per90",
        "npxg_per90",
        "average_npxg_per_shot",

        "final_third_receipts_per90",
        "box_receipts_per90",
        "receipts_under_pressure_per90",

        "average_on_ball_x",
        "average_on_ball_width",
        "average_defensive_x",
        "average_defensive_width",
    ],

    "FW": [
        "pass_attempts_per90",
        "pass_completion_pct",
        "progressive_passes_per90",
        "passes_into_box_per90",
        "through_balls_per90",
        "xa_per90",

        "average_carry_length",
        "progressive_carries_per90",
        "carries_into_box_per90",
        "dribbles_per90",
        "dispossessed_per90",
        "miscontrols_per90",

        "pressures_per90",
        "ball_recoveries_per90",

        "non_penalty_shots_per90",
        "npxg_per90",
        "average_npxg_per_shot",

        "final_third_receipts_per90",
        "box_receipts_per90",
        "receipts_under_pressure_per90",

        "average_on_ball_x",
        "average_on_ball_width",
        "average_defensive_x",
        "average_defensive_width",
    ],
}


# ======================================================
# SPLIT BUILDING
# ======================================================

def build_split_profiles(
    match_profiles: pd.DataFrame,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Randomly split each player's matches
    into two independent subsets.
    """

    rng = np.random.default_rng(
        seed
    )

    split_parts = []

    grouped = match_profiles.groupby(
        PLAYER_KEYS,
        sort=True,
        dropna=False,
    )

    for _, group in grouped:
        group = group.copy()

        n_matches = len(group)

        if n_matches < 2:
            continue

        order = np.arange(
            n_matches
        )

        rng.shuffle(
            order
        )

        n_a = n_matches // 2

        labels = np.full(
            n_matches,
            "B",
            dtype=object,
        )

        labels[
            order[:n_a]
        ] = "A"

        group[
            "_split"
        ] = labels

        split_parts.append(
            group
        )

    split_data = pd.concat(
        split_parts,
        ignore_index=True,
    )

    a_matches = (
        split_data[
            split_data[
                "_split"
            ] == "A"
        ]
        .drop(
            columns=[
                "_split",
            ]
        )
    )

    b_matches = (
        split_data[
            split_data[
                "_split"
            ] == "B"
        ]
        .drop(
            columns=[
                "_split",
            ]
        )
    )

    profiles_a = (
        aggregate_player_profiles(
            a_matches
        )
    )

    profiles_b = (
        aggregate_player_profiles(
            b_matches
        )
    )

    return (
        profiles_a,
        profiles_b,
    )


def restrict_to_valid_pairs(
    profiles_a: pd.DataFrame,
    profiles_b: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Keep only players present in both halves
    with enough minutes in each half.
    """

    minutes_a = profiles_a[
        PLAYER_KEYS
        + [
            "minutes_played",
        ]
    ].rename(
        columns={
            "minutes_played":
                "minutes_a",
        }
    )

    minutes_b = profiles_b[
        PLAYER_KEYS
        + [
            "minutes_played",
        ]
    ].rename(
        columns={
            "minutes_played":
                "minutes_b",
        }
    )

    valid = minutes_a.merge(
        minutes_b,
        on=PLAYER_KEYS,
        how="inner",
    )

    valid = valid[
        (
            valid[
                "minutes_a"
            ]
            >= MIN_SPLIT_MINUTES
        )
        & (
            valid[
                "minutes_b"
            ]
            >= MIN_SPLIT_MINUTES
        )
    ]

    valid_keys = valid[
        PLAYER_KEYS
    ]

    profiles_a = (
        profiles_a
        .merge(
            valid_keys,
            on=PLAYER_KEYS,
            how="inner",
        )
        .reset_index(
            drop=True
        )
    )

    profiles_b = (
        profiles_b
        .merge(
            valid_keys,
            on=PLAYER_KEYS,
            how="inner",
        )
        .reset_index(
            drop=True
        )
    )

    return (
        profiles_a,
        profiles_b,
    )


# ======================================================
# SIMILARITY EVALUATION
# ======================================================

def player_identity(
    row: pd.Series,
) -> tuple:
    return (
        row[
            "player.id"
        ],
        row[
            "team.name"
        ],
    )


def directional_ranks(
    query_profiles: pd.DataFrame,
    candidate_profiles: pd.DataFrame,
    features: list[str],
) -> list[int]:
    """
    Search candidate profiles for the same
    player represented by the query profile.

    Scaler is fitted only on candidate data.
    """

    candidate_values = (
        candidate_profiles[
            features
        ]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    query_values = (
        query_profiles[
            features
        ]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    medians = (
        candidate_values
        .median()
        .fillna(0.0)
    )

    candidate_values = (
        candidate_values
        .fillna(
            medians
        )
    )

    query_values = (
        query_values
        .fillna(
            medians
        )
    )

    scaler = StandardScaler()

    candidate_scaled = (
        scaler.fit_transform(
            candidate_values
        )
    )

    query_scaled = (
        scaler.transform(
            query_values
        )
    )

    similarities = cosine_similarity(
        query_scaled,
        candidate_scaled,
    )

    candidate_lookup = {
        player_identity(row):
            index
        for index, row
        in candidate_profiles.iterrows()
    }

    ranks = []

    for query_index, row in (
        query_profiles.iterrows()
    ):
        identity = player_identity(
            row
        )

        target_index = (
            candidate_lookup.get(
                identity
            )
        )

        if target_index is None:
            continue

        scores = similarities[
            query_index
        ]

        order = np.argsort(
            -scores
        )

        rank_positions = np.where(
            order
            == target_index
        )[0]

        if len(rank_positions) == 0:
            continue

        rank = int(
            rank_positions[0]
        ) + 1

        ranks.append(
            rank
        )

    return ranks


def evaluate_model(
    profiles_a: pd.DataFrame,
    profiles_b: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """
    Evaluate both directions:
    A -> B and B -> A.
    """

    ranks_ab = directional_ranks(
        profiles_a,
        profiles_b,
        features,
    )

    ranks_ba = directional_ranks(
        profiles_b,
        profiles_a,
        features,
    )

    ranks = np.asarray(
        ranks_ab
        + ranks_ba,
        dtype=float,
    )

    if len(ranks) == 0:
        return {
            "queries": 0,
            "top1": np.nan,
            "top3": np.nan,
            "top5": np.nan,
            "median_rank": np.nan,
            "mrr": np.nan,
        }

    return {
        "queries":
            len(ranks),

        "top1":
            float(
                np.mean(
                    ranks <= 1
                )
            ),

        "top3":
            float(
                np.mean(
                    ranks <= 3
                )
            ),

        "top5":
            float(
                np.mean(
                    ranks <= 5
                )
            ),

        "median_rank":
            float(
                np.median(
                    ranks
                )
            ),

        "mrr":
            float(
                np.mean(
                    1.0 / ranks
                )
            ),
    }


# ======================================================
# MAIN
# ======================================================

def main() -> None:
    match_profiles = (
        pd.read_parquet(
            MATCH_PROFILES_PATH
        )
    )

    season_profiles = (
        pd.read_parquet(
            SEASON_PROFILES_PATH
        )
    )

    print("=" * 90)
    print(
        "PREMIER LEAGUE "
        "ROLE MODEL EVALUATION"
    )
    print("=" * 90)

    print(
        f"Repeats: "
        f"{N_REPEATS}"
    )

    print(
        f"Minimum season minutes: "
        f"{MIN_SEASON_MINUTES}"
    )

    print(
        f"Minimum split minutes: "
        f"{MIN_SPLIT_MINUTES}"
    )

    all_results = []

    for role_index, role in enumerate(
        ROLE_GROUPS
    ):
        print()
        print("=" * 90)
        print(
            f"ROLE: {role}"
        )
        print("=" * 90)

        eligible = season_profiles[
            (
                season_profiles[
                    "position_group"
                ] == role
            )
            & (
                season_profiles[
                    "minutes_played"
                ]
                >= MIN_SEASON_MINUTES
            )
            & (
                season_profiles[
                    "matches_played"
                ]
                >= MIN_MATCHES
            )
        ].copy()

        print(
            f"Season-eligible players: "
            f"{len(eligible)}"
        )

        eligible_keys = eligible[
            PLAYER_KEYS
        ]

        role_matches = (
            match_profiles
            .merge(
                eligible_keys,
                on=PLAYER_KEYS,
                how="inner",
            )
        )

        model_features = {
            "LEGACY_27":
                LEGACY_FEATURES,

            "FULL_51":
                FULL_FEATURES,

            "REDUCED_34":
                REDUCED_FEATURES,

            "ROLE_AWARE":
                ROLE_FEATURES[
                    role
                ],
        }

        for repeat in range(
            N_REPEATS
        ):
            seed = (
                BASE_SEED
                + repeat
                + role_index * 10000
            )

            (
                profiles_a,
                profiles_b,
            ) = build_split_profiles(
                role_matches,
                seed,
            )

            (
                profiles_a,
                profiles_b,
            ) = restrict_to_valid_pairs(
                profiles_a,
                profiles_b,
            )

            n_players = len(
                profiles_a
            )

            if n_players < 6:
                continue

            for (
                model_name,
                features,
            ) in model_features.items():

                missing = [
                    feature
                    for feature in features
                    if (
                        feature
                        not in profiles_a.columns
                        or feature
                        not in profiles_b.columns
                    )
                ]

                if missing:
                    raise ValueError(
                        f"{role} / "
                        f"{model_name} "
                        f"missing features: "
                        f"{missing}"
                    )

                metrics = (
                    evaluate_model(
                        profiles_a,
                        profiles_b,
                        features,
                    )
                )

                all_results.append(
                    {
                        "role":
                            role,

                        "repeat":
                            repeat,

                        "seed":
                            seed,

                        "model":
                            model_name,

                        "n_features":
                            len(
                                features
                            ),

                        "eligible_players":
                            n_players,

                        **metrics,
                    }
                )

        print(
            "Finished."
        )

    results = pd.DataFrame(
        all_results
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    summary = (
        results
        .groupby(
            [
                "role",
                "model",
                "n_features",
            ],
            as_index=False,
        )
        .agg(
            players_mean=(
                "eligible_players",
                "mean",
            ),

            top1_mean=(
                "top1",
                "mean",
            ),
            top1_std=(
                "top1",
                "std",
            ),

            top3_mean=(
                "top3",
                "mean",
            ),
            top3_std=(
                "top3",
                "std",
            ),

            top5_mean=(
                "top5",
                "mean",
            ),
            top5_std=(
                "top5",
                "std",
            ),

            median_rank_mean=(
                "median_rank",
                "mean",
            ),

            mrr_mean=(
                "mrr",
                "mean",
            ),
            mrr_std=(
                "mrr",
                "std",
            ),
        )
    )

    for role in ROLE_GROUPS:
        print()
        print(
            f"ROLE: {role}"
        )
        print(
            "-" * 90
        )

        role_summary = summary[
            summary[
                "role"
            ] == role
        ].copy()

        role_summary = (
            role_summary
            .sort_values(
                [
                    "top5_mean",
                    "mrr_mean",
                ],
                ascending=False,
            )
        )

        print(
            role_summary[
                [
                    "model",
                    "n_features",
                    "players_mean",
                    "top1_mean",
                    "top3_mean",
                    "top5_mean",
                    "median_rank_mean",
                    "mrr_mean",
                ]
            ]
            .to_string(
                index=False,
                float_format=lambda x:
                f"{x:.3f}",
            )
        )

    print()
    print(
        "Saved detailed results to:"
    )
    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()