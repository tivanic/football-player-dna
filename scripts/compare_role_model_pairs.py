from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_"
    "role_model_evaluation.csv"
)


COMPARISONS = [
    (
        "CB",
        "ROLE_AWARE",
        "FULL_51",
    ),
    (
        "FB",
        "ROLE_AWARE",
        "FULL_51",
    ),
    (
        "DM_CM",
        "FULL_51",
        "REDUCED_34",
    ),
    (
        "AM_W",
        "FULL_51",
        "REDUCED_34",
    ),
    (
        "FW",
        "FULL_51",
        "REDUCED_34",
    ),
]


METRICS = [
    "top1",
    "top3",
    "top5",
    "mrr",
]


N_BOOTSTRAP = 10000
RANDOM_SEED = 42


def bootstrap_ci(
    values: np.ndarray,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for the mean paired
    difference between two models.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    n = len(values)

    bootstrap_means = np.empty(
        N_BOOTSTRAP
    )

    for i in range(
        N_BOOTSTRAP
    ):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        bootstrap_means[i] = (
            sample.mean()
        )

    lower = np.percentile(
        bootstrap_means,
        2.5,
    )

    upper = np.percentile(
        bootstrap_means,
        97.5,
    )

    return (
        float(lower),
        float(upper),
    )


def main() -> None:
    results = pd.read_csv(
        RESULTS_PATH
    )

    print("=" * 90)
    print(
        "PAIRED MODEL COMPARISON"
    )
    print("=" * 90)

    for (
        role,
        model_a,
        model_b,
    ) in COMPARISONS:

        role_data = results[
            results[
                "role"
            ] == role
        ]

        a = (
            role_data[
                role_data[
                    "model"
                ] == model_a
            ]
            .set_index(
                "repeat"
            )
        )

        b = (
            role_data[
                role_data[
                    "model"
                ] == model_b
            ]
            .set_index(
                "repeat"
            )
        )

        common_repeats = (
            a.index
            .intersection(
                b.index
            )
        )

        print()
        print("=" * 90)

        print(
            f"{role}: "
            f"{model_a} "
            f"vs "
            f"{model_b}"
        )

        print("=" * 90)

        print(
            f"Paired repeats: "
            f"{len(common_repeats)}"
        )

        for metric in METRICS:

            values_a = (
                a.loc[
                    common_repeats,
                    metric,
                ]
                .to_numpy(
                    dtype=float
                )
            )

            values_b = (
                b.loc[
                    common_repeats,
                    metric,
                ]
                .to_numpy(
                    dtype=float
                )
            )

            difference = (
                values_a
                - values_b
            )

            (
                ci_low,
                ci_high,
            ) = bootstrap_ci(
                difference
            )

            tolerance = 1e-12

            wins = np.mean(
                difference
                > tolerance
            )

            losses = np.mean(
                difference
                < -tolerance
            )

            ties = np.mean(
                np.abs(
                    difference
                )
                <= tolerance
            )

            print()

            print(
                metric.upper()
            )

            print(
                f"  {model_a}: "
                f"{values_a.mean():.4f}"
            )

            print(
                f"  {model_b}: "
                f"{values_b.mean():.4f}"
            )

            print(
                f"  Mean delta: "
                f"{difference.mean():+.4f}"
            )

            print(
                f"  95% bootstrap CI: "
                f"[{ci_low:+.4f}, "
                f"{ci_high:+.4f}]"
            )

            print(
                f"  {model_a} wins: "
                f"{wins * 100:.1f}%"
            )

            print(
                f"  Ties: "
                f"{ties * 100:.1f}%"
            )

            print(
                f"  {model_b} wins: "
                f"{losses * 100:.1f}%"
            )


if __name__ == "__main__":
    main()