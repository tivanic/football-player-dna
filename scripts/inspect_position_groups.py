from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_player_profiles.parquet"
)


def main() -> None:
    profiles = pd.read_parquet(
        DATA_PATH
    )

    print("POSITION GROUP COUNTS")
    print("---------------------")

    print(
        profiles["position_group"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("OTHER PRIMARY POSITIONS")
    print("-----------------------")

    other = profiles[
        profiles["position_group"] == "OTHER"
    ]

    position_counts = (
        other["primary_position"]
        .value_counts(dropna=False)
    )

    print(
        position_counts.to_string()
    )

    print()
    print("OTHER PLAYERS")
    print("-------------")

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
            [
                "primary_position",
                "minutes_played",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()