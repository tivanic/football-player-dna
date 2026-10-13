from pathlib import Path

import pandas as pd

from player_dna.features.competition_profile import (
    aggregate_player_profiles,
)


MATCH_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_match_profiles.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_player_profiles.parquet"
)


def main() -> None:
    print(
        f"Loading match profiles from:"
    )
    print(MATCH_PATH)

    match_profiles = pd.read_parquet(
        MATCH_PATH
    )

    print(
        f"\nLoaded "
        f"{len(match_profiles)} "
        f"match-level player profiles."
    )

    player_profiles = (
        aggregate_player_profiles(
            match_profiles
        )
    )

    player_profiles.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved "
        f"{len(player_profiles)} "
        f"competition player profiles to:"
    )
    print(OUTPUT_PATH)

    print()
    print("POSITION GROUPS")
    print("---------------")

    print(
        player_profiles[
            "position_group"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("REMAINING OTHER PLAYERS")
    print("-----------------------")

    other = player_profiles[
        player_profiles[
            "position_group"
        ] == "OTHER"
    ]

    if other.empty:
        print("None")
        return

    columns = [
        "player.name",
        "team.name",
        "primary_position",
        "matches_played",
        "minutes_played",
    ]

    print(
        other[
            columns
        ]
        .sort_values(
            "minutes_played",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()