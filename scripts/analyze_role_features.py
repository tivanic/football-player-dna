from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_player_profiles.parquet"
)

MIN_MINUTES = 900

ROLE_GROUPS = [
    "CB",
    "FB",
    "DM_CM",
    "AM_W",
    "FW",
]

CORRELATION_THRESHOLD = 0.80


CANDIDATE_FEATURES = [
    # ==================================================
    # Distribution / passing
    # ==================================================
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

    # ==================================================
    # Carrying / possession
    # ==================================================
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

    # ==================================================
    # Defending
    # ==================================================
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

    # ==================================================
    # Attacking
    # ==================================================
    "non_penalty_shots_per90",
    "non_penalty_shots_in_box_per90",
    "npxg_per90",
    "average_npxg_per_shot",

    # ==================================================
    # Creation
    # ==================================================
    "xa_per90",

    # ==================================================
    # Receiving
    # ==================================================
    "final_third_receipts_per90",
    "box_receipts_per90",
    "receipts_under_pressure_per90",

    # ==================================================
    # Spatial
    # ==================================================
    "average_on_ball_x",
    "average_on_ball_width",
    "average_receipt_x",
    "average_receipt_width",
    "average_defensive_x",
    "average_defensive_width",
    "attacking_third_action_share",
    "wide_action_share",
]


def build_feature_summary(
    data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []

    for feature in features:
        series = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

        finite = series.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        non_missing = finite.dropna()

        missing_pct = (
            finite.isna().mean()
            * 100
        )

        if len(non_missing) > 0:
            zero_pct = (
                (non_missing == 0)
                .mean()
                * 100
            )

            mean = non_missing.mean()
            std = non_missing.std()
            minimum = non_missing.min()
            maximum = non_missing.max()

        else:
            zero_pct = np.nan
            mean = np.nan
            std = np.nan
            minimum = np.nan
            maximum = np.nan

        rows.append(
            {
                "feature": feature,
                "missing_pct": missing_pct,
                "zero_pct": zero_pct,
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
            }
        )

    return pd.DataFrame(
        rows
    )


def find_high_correlations(
    data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    numeric = (
        data[
            features
        ]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    correlations = numeric.corr()

    rows = []

    for i, feature_a in enumerate(
        features
    ):
        for feature_b in features[
            i + 1:
        ]:
            value = correlations.loc[
                feature_a,
                feature_b,
            ]

            if pd.isna(value):
                continue

            if (
                abs(value)
                >= CORRELATION_THRESHOLD
            ):
                rows.append(
                    {
                        "feature_a":
                            feature_a,
                        "feature_b":
                            feature_b,
                        "correlation":
                            value,
                        "abs_correlation":
                            abs(value),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "correlation",
                "abs_correlation",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "abs_correlation",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    profiles = pd.read_parquet(
        DATA_PATH
    )

    available_features = [
        feature
        for feature in CANDIDATE_FEATURES
        if feature in profiles.columns
    ]

    missing_features = [
        feature
        for feature in CANDIDATE_FEATURES
        if feature not in profiles.columns
    ]

    print("=" * 90)
    print("PLAYER DNA ROLE FEATURE AUDIT")
    print("=" * 90)

    print(
        f"Dataset players: "
        f"{len(profiles)}"
    )

    print(
        f"Minimum minutes: "
        f"{MIN_MINUTES}"
    )

    print(
        f"Candidate features available: "
        f"{len(available_features)}"
    )

    if missing_features:
        print()
        print("WARNING - MISSING CANDIDATE FEATURES")
        print("------------------------------------")

        for feature in missing_features:
            print(feature)

    output_rows = []

    for role in ROLE_GROUPS:
        role_data = profiles[
            (
                profiles[
                    "position_group"
                ] == role
            )
            & (
                profiles[
                    "minutes_played"
                ] >= MIN_MINUTES
            )
        ].copy()

        print()
        print()
        print("=" * 90)
        print(
            f"ROLE: {role}"
        )
        print("=" * 90)

        print(
            f"Eligible players: "
            f"{len(role_data)}"
        )

        if role_data.empty:
            continue

        summary = build_feature_summary(
            role_data,
            available_features,
        )

        summary[
            "role"
        ] = role

        output_rows.append(
            summary
        )

        # --------------------------------------------------
        # Potentially problematic features
        # --------------------------------------------------

        problematic = summary[
            (
                summary[
                    "missing_pct"
                ] >= 30
            )
            |
            (
                summary[
                    "zero_pct"
                ] >= 75
            )
            |
            (
                summary[
                    "std"
                ].fillna(0)
                < 1e-8
            )
        ].copy()

        print()
        print(
            "POTENTIALLY WEAK / SPARSE FEATURES"
        )
        print(
            "----------------------------------"
        )

        if problematic.empty:
            print("None")
        else:
            print(
                problematic[
                    [
                        "feature",
                        "missing_pct",
                        "zero_pct",
                        "std",
                    ]
                ]
                .sort_values(
                    [
                        "missing_pct",
                        "zero_pct",
                    ],
                    ascending=False,
                )
                .to_string(
                    index=False,
                    float_format=lambda x:
                    f"{x:.3f}",
                )
            )

        # --------------------------------------------------
        # Correlations
        # --------------------------------------------------

        high_corr = (
            find_high_correlations(
                role_data,
                available_features,
            )
        )

        print()
        print(
            f"HIGH CORRELATIONS "
            f"(|r| >= "
            f"{CORRELATION_THRESHOLD:.2f})"
        )
        print(
            "----------------------------------"
        )

        if high_corr.empty:
            print("None")
        else:
            print(
                high_corr[
                    [
                        "feature_a",
                        "feature_b",
                        "correlation",
                    ]
                ]
                .to_string(
                    index=False,
                    float_format=lambda x:
                    f"{x:.3f}",
                )
            )

    # --------------------------------------------------
    # Save audit table
    # --------------------------------------------------

    if output_rows:
        audit = pd.concat(
            output_rows,
            ignore_index=True,
        )

        output_path = Path(
            "data/processed/"
            "premier_league_2015_2016_"
            "role_feature_audit.csv"
        )

        audit.to_csv(
            output_path,
            index=False,
        )

        print()
        print("=" * 90)

        print(
            "Saved role feature audit to:"
        )

        print(
            output_path
        )


if __name__ == "__main__":
    main()