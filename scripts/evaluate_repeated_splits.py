from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from player_dna.features.competition_profile import (
    aggregate_player_profiles,
)

from player_dna.model.similarity import (
    PASSING_DNA_FEATURES,
    POSSESSION_DNA_FEATURES,
    PLAYER_DNA_FEATURES,
    REDUCED_PLAYER_DNA_FEATURES,
)


MATCH_PATH = Path(
    "data/processed/world_cup_2022_match_profiles.parquet"
)

FULL_PROFILE_PATH = Path(
    "data/processed/world_cup_2022_player_profiles.parquet"
)

POSITION_GROUP = "DM_CM"

MIN_MATCHES = 4
MIN_SPLIT_MINUTES = 120

N_REPEATS = 100
RANDOM_SEED = 42


MODEL_VERSIONS = {
    "v0.1 Passing": (
        PASSING_DNA_FEATURES
    ),
    "v0.2 Passing + Possession": (
        PASSING_DNA_FEATURES
        + POSSESSION_DNA_FEATURES
    ),
    "v0.3 Full": (
        PLAYER_DNA_FEATURES
    ),
    "v0.3 Reduced": (
        REDUCED_PLAYER_DNA_FEATURES
    ),
}


def build_random_split_profiles(
    match_profiles: pd.DataFrame,
    full_profiles: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split each eligible player's matches
    into two groups and aggregate each group
    into a separate Player DNA profile.
    """

    eligible = full_profiles[
        (full_profiles["position_group"] == POSITION_GROUP)
        & (
            full_profiles["matches_played"]
            >= MIN_MATCHES
        )
    ].copy()

    eligible_ids = set(
        eligible["player.id"]
    )

    data = match_profiles[
        match_profiles["player.id"].isin(
            eligible_ids
        )
    ].copy()

    rng = np.random.default_rng(
        random_state
    )

    split_labels = []

    for _, group in data.groupby(
        [
            "player.id",
            "team.name",
        ]
    ):
        indices = (
            group.index
            .to_numpy()
            .copy()
        )

        rng.shuffle(indices)

        midpoint = len(indices) // 2

        a_indices = set(
            indices[:midpoint]
        )

        for index in group.index:
            if index in a_indices:
                split_labels.append(
                    (index, "A")
                )
            else:
                split_labels.append(
                    (index, "B")
                )

    split_map = dict(
        split_labels
    )

    data["split"] = (
        data.index.map(
            split_map
        )
    )

    split_a = aggregate_player_profiles(
        data[
            data["split"] == "A"
        ]
    )

    split_b = aggregate_player_profiles(
        data[
            data["split"] == "B"
        ]
    )

    # Require a minimum amount of playing time
    # in each half of the split.
    split_a = split_a[
        split_a["minutes_played"]
        >= MIN_SPLIT_MINUTES
    ].copy()

    split_b = split_b[
        split_b["minutes_played"]
        >= MIN_SPLIT_MINUTES
    ].copy()

    # Only keep players who survived the minutes
    # threshold in both A and B.
    common_ids = set(
        split_a["player.id"]
    ).intersection(
        split_b["player.id"]
    )

    split_a = (
        split_a[
            split_a["player.id"].isin(
                common_ids
            )
        ]
        .reset_index(drop=True)
    )

    split_b = (
        split_b[
            split_b["player.id"].isin(
                common_ids
            )
        ]
        .reset_index(drop=True)
    )

    return split_a, split_b


def get_self_ranks(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    features: list[str],
) -> list[int]:
    """
    Rank every candidate profile by similarity
    to each query profile.

    Return the rank at which the same player
    appears in the opposite split.
    """

    query_features = (
        queries[features]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    candidate_features = (
        candidates[features]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # Calculate medians using the candidate set.
    # The same values are then used to fill
    # missing data in queries.
    medians = (
        candidate_features
        .median()
    )

    candidate_features = (
        candidate_features
        .fillna(medians)
    )

    query_features = (
        query_features
        .fillna(medians)
    )

    # Fit scaler only on the candidate set.
    scaler = StandardScaler()

    candidate_scaled = (
        scaler.fit_transform(
            candidate_features
        )
    )

    query_scaled = (
        scaler.transform(
            query_features
        )
    )

    similarities = cosine_similarity(
        query_scaled,
        candidate_scaled,
    )

    candidate_ids = (
        candidates["player.id"]
        .to_numpy()
    )

    ranks = []

    for query_index, query in queries.iterrows():

        scores = similarities[
            query_index
        ]

        ranking = np.argsort(
            -scores
        )

        ranked_ids = (
            candidate_ids[
                ranking
            ]
        )

        positions = np.where(
            ranked_ids
            == query["player.id"]
        )[0]

        if len(positions) == 0:
            continue

        self_rank = (
            int(positions[0]) + 1
        )

        ranks.append(
            self_rank
        )

    return ranks


def calculate_metrics(
    ranks: list[int],
) -> dict[str, float]:
    """
    Calculate retrieval metrics from
    self-match ranks.
    """

    ranks_array = np.asarray(
        ranks,
        dtype=float,
    )

    return {
        "top1":
            np.mean(
                ranks_array <= 1
            ),

        "top3":
            np.mean(
                ranks_array <= 3
            ),

        "top5":
            np.mean(
                ranks_array <= 5
            ),

        "median_rank":
            np.median(
                ranks_array
            ),

        "mrr":
            np.mean(
                1 / ranks_array
            ),
    }


def main() -> None:
    match_profiles = pd.read_parquet(
        MATCH_PATH
    )

    full_profiles = pd.read_parquet(
        FULL_PROFILE_PATH
    )

    rows = []

    for repeat in range(
        N_REPEATS
    ):
        seed = (
            RANDOM_SEED
            + repeat
        )

        split_a, split_b = (
            build_random_split_profiles(
                match_profiles=match_profiles,
                full_profiles=full_profiles,
                random_state=seed,
            )
        )

        # IMPORTANT:
        # Every model version uses exactly
        # the same A/B split for this repeat.
        for (
            model_name,
            features,
        ) in MODEL_VERSIONS.items():

            ranks_a_to_b = (
                get_self_ranks(
                    queries=split_a,
                    candidates=split_b,
                    features=features,
                )
            )

            ranks_b_to_a = (
                get_self_ranks(
                    queries=split_b,
                    candidates=split_a,
                    features=features,
                )
            )

            ranks = (
                ranks_a_to_b
                + ranks_b_to_a
            )

            metrics = (
                calculate_metrics(
                    ranks
                )
            )

            metrics["repeat"] = repeat
            metrics["model"] = model_name
            metrics["players"] = len(
                split_a
            )

            rows.append(
                metrics
            )

        if (
            repeat + 1
        ) % 10 == 0:
            print(
                f"Completed "
                f"{repeat + 1}/{N_REPEATS} "
                f"splits..."
            )

    results = pd.DataFrame(
        rows
    )

    print()
    print(
        f"Repeated splits: "
        f"{N_REPEATS}"
    )

    # The number of eligible players is identical
    # across models within the same repeat, so use
    # one model when reporting the average.
    reference_model = next(
        iter(MODEL_VERSIONS)
    )

    reference_results = results[
        results["model"]
        == reference_model
    ]

    print(
        f"Average eligible players: "
        f"{reference_results['players'].mean():.1f}"
    )

    print()

    for model_name in MODEL_VERSIONS:

        model_results = results[
            results["model"]
            == model_name
        ]

        print("=" * 80)
        print(model_name)

        print(
            f"Features: "
            f"{len(MODEL_VERSIONS[model_name])}"
        )

        print("=" * 80)

        for metric in [
            "top1",
            "top3",
            "top5",
            "mrr",
        ]:
            mean = (
                model_results[
                    metric
                ]
                .mean()
            )

            std = (
                model_results[
                    metric
                ]
                .std()
            )

            print(
                f"{metric:12s} "
                f"{mean:.3f} "
                f"± "
                f"{std:.3f}"
            )

        median_mean = (
            model_results[
                "median_rank"
            ]
            .mean()
        )

        median_std = (
            model_results[
                "median_rank"
            ]
            .std()
        )

        print(
            f"{'median_rank':12s} "
            f"{median_mean:.2f} "
            f"± "
            f"{median_std:.2f}"
        )

        print()


if __name__ == "__main__":
    main()