from pathlib import Path

import pandas as pd

from player_dna.model.player_dna_v1 import (
    MODEL_FEATURES,
    MODEL_NAMES,
    find_similar_players_v1,
)


DATA_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_player_profiles.parquet"
)


def test_selected_model_feature_counts():
    assert len(MODEL_FEATURES["CB"]) == 20
    assert len(MODEL_FEATURES["FB"]) == 30
    assert len(MODEL_FEATURES["DM_CM"]) == 51
    assert len(MODEL_FEATURES["AM_W"]) == 34
    assert len(MODEL_FEATURES["FW"]) == 34


def test_selected_model_names():
    assert MODEL_NAMES == {
        "CB": "ROLE_AWARE_20",
        "FB": "ROLE_AWARE_30",
        "DM_CM": "FULL_51",
        "AM_W": "REDUCED_34",
        "FW": "REDUCED_34",
    }


def test_all_model_features_exist_in_demo_data():
    profiles = pd.read_parquet(
        DATA_PATH
    )

    required_features = {
        feature
        for features in MODEL_FEATURES.values()
        for feature in features
    }

    missing = (
        required_features
        - set(profiles.columns)
    )

    assert not missing


def test_similarity_excludes_target_and_stays_in_role():
    profiles = pd.read_parquet(
        DATA_PATH
    )

    target, results = (
        find_similar_players_v1(
            profiles,
            player_name="Mesut Özil",
            team_name="Arsenal",
            min_minutes=900,
            top_n=5,
        )
    )

    assert len(results) == 5

    assert (
        results["position_group"]
        == target["position_group"]
    ).all()

    same_player = (
        (
            results["player.id"]
            == target["player.id"]
        )
        & (
            results["team.name"]
            == target["team.name"]
        )
    )

    assert not same_player.any()


def test_similarity_results_are_ranked():
    profiles = pd.read_parquet(
        DATA_PATH
    )

    _, results = (
        find_similar_players_v1(
            profiles,
            player_name="Kanté",
            team_name="Leicester City",
            min_minutes=900,
            top_n=5,
        )
    )

    assert results["rank"].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]

    similarities = (
        results[
            "similarity_score"
        ]
        .tolist()
    )

    assert similarities == sorted(
        similarities,
        reverse=True,
    )