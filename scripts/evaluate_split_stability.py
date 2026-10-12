from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from player_dna.features.competition_profile import (
    aggregate_player_profiles,
)

# from player_dna.model.similarity import (
#     PLAYER_DNA_FEATURES,
#     REDUCED_PLAYER_DNA_FEATURES,
# )
from player_dna.model.similarity import (
    PASSING_DNA_FEATURES,
    POSSESSION_DNA_FEATURES,
    DEFENDING_DNA_FEATURES,
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


def build_split_profiles(
    match_profiles: pd.DataFrame,
    full_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    eligible = full_profiles[
        (full_profiles["position_group"] == POSITION_GROUP)
        & (full_profiles["matches_played"] >= MIN_MATCHES)
    ].copy()

    eligible_ids = set(
        eligible["player.id"]
    )

    data = match_profiles[
        match_profiles["player.id"].isin(
            eligible_ids
        )
    ].copy()

    data["match_date"] = pd.to_datetime(
        data["match_date"],
        errors="coerce",
    )

    data = data.sort_values(
        [
            "player.id",
            "team.name",
            "match_date",
            "match_id",
        ]
    )

    data["player_match_number"] = (
        data
        .groupby(
            [
                "player.id",
                "team.name",
            ]
        )
        .cumcount()
    )

    # Alternating matches:
    # 0, 2, 4... -> A
    # 1, 3, 5... -> B
    data["split"] = np.where(
        data["player_match_number"] % 2 == 0,
        "A",
        "B",
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

    split_a = split_a[
        split_a["minutes_played"]
        >= MIN_SPLIT_MINUTES
    ].copy()

    split_b = split_b[
        split_b["minutes_played"]
        >= MIN_SPLIT_MINUTES
    ].copy()

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


def evaluate_direction(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    features: list[str],
    direction: str,
) -> pd.DataFrame:

    queries = queries.reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)

    candidate_features = (
        candidates[features]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    query_features = (
        queries[features]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    medians = candidate_features.median()

    candidate_features = (
        candidate_features.fillna(medians)
    )

    query_features = (
        query_features.fillna(medians)
    )

    scaler = StandardScaler()

    candidate_scaled = scaler.fit_transform(
        candidate_features
    )

    query_scaled = scaler.transform(
        query_features
    )

    similarities = cosine_similarity(
        query_scaled,
        candidate_scaled,
    )

    candidate_ids = (
        candidates["player.id"]
        .to_numpy()
    )

    rows = []

    for query_index, query in queries.iterrows():

        scores = similarities[
            query_index
        ]

        ranking = np.argsort(
            -scores
        )

        ranked_player_ids = (
            candidate_ids[ranking]
        )

        self_positions = np.where(
            ranked_player_ids
            == query["player.id"]
        )[0]

        if len(self_positions) == 0:
            continue

        self_rank = (
            int(self_positions[0]) + 1
        )

        candidate_index = ranking[
            self_rank - 1
        ]

        self_similarity = scores[
            candidate_index
        ]

        rows.append(
            {
                "player.id":
                    query["player.id"],
                "player.name":
                    query["player.name"],
                "direction":
                    direction,
                "self_rank":
                    self_rank,
                "self_similarity":
                    self_similarity,
            }
        )

    return pd.DataFrame(rows)


def print_metrics(
    results: pd.DataFrame,
) -> None:

    print(
        f"Comparisons: {len(results)}"
    )

    print(
        f"Top-1 accuracy: "
        f"{(results['self_rank'] <= 1).mean():.1%}"
    )

    print(
        f"Top-3 accuracy: "
        f"{(results['self_rank'] <= 3).mean():.1%}"
    )

    print(
        f"Top-5 accuracy: "
        f"{(results['self_rank'] <= 5).mean():.1%}"
    )

    print(
        f"Median self-rank: "
        f"{results['self_rank'].median():.1f}"
    )

    mean_reciprocal_rank = (
        (1 / results["self_rank"])
        .mean()
    )

    print(
        f"Mean reciprocal rank: "
        f"{mean_reciprocal_rank:.3f}"
    )


def evaluate_model(
    name: str,
    features: list[str],
    split_a: pd.DataFrame,
    split_b: pd.DataFrame,
) -> None:

    print()
    print("=" * 80)
    print(name)
    print(f"Features: {len(features)}")
    print("=" * 80)

    a_to_b = evaluate_direction(
        queries=split_a,
        candidates=split_b,
        features=features,
        direction="A -> B",
    )

    b_to_a = evaluate_direction(
        queries=split_b,
        candidates=split_a,
        features=features,
        direction="B -> A",
    )

    results = pd.concat(
        [
            a_to_b,
            b_to_a,
        ],
        ignore_index=True,
    )

    print_metrics(results)

    print(
        "\nBEST SELF-MATCHES"
    )

    print(
        results
        .sort_values(
            "self_rank"
        )
        .head(10)
        .to_string(
            index=False,
            formatters={
                "self_similarity":
                    lambda value:
                    f"{value:.3f}",
            },
        )
    )

    print(
        "\nWORST SELF-MATCHES"
    )

    print(
        results
        .sort_values(
            "self_rank",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False,
            formatters={
                "self_similarity":
                    lambda value:
                    f"{value:.3f}",
            },
        )
    )


def main() -> None:

    match_profiles = pd.read_parquet(
        MATCH_PATH
    )

    full_profiles = pd.read_parquet(
        FULL_PROFILE_PATH
    )

    split_a, split_b = (
        build_split_profiles(
            match_profiles,
            full_profiles,
        )
    )

    print(
        f"Eligible split players: "
        f"{len(split_a)}"
    )
    #****
    evaluate_model(
        name="v0.1 Passing",
        features=PASSING_DNA_FEATURES,
        split_a=split_a,
        split_b=split_b,
    )

    evaluate_model(
        name="v0.2 Passing + Possession",
        features=(
            PASSING_DNA_FEATURES
            + POSSESSION_DNA_FEATURES
        ),
        split_a=split_a,
        split_b=split_b,
    )

    evaluate_model(
        name="v0.3 Full",
        features=PLAYER_DNA_FEATURES,
        split_a=split_a,
        split_b=split_b,
    )

    evaluate_model(
        name="v0.3 Reduced",
        features=REDUCED_PLAYER_DNA_FEATURES,
        split_a=split_a,
        split_b=split_b,
    )
    #****


if __name__ == "__main__":
    main()