from __future__ import annotations

from player_dna.data.statsbomb import (
    load_events,
)

from player_dna.features.attacking import (
    build_attacking_features,
)


MATCH_ID = 3754217


def main() -> None:
    events = load_events(
        MATCH_ID
    )

    attacking = (
        build_attacking_features(
            events
        )
    )

    print(
        "ATTACKING FEATURES"
    )
    print(
        "------------------"
    )

    columns = [
        "player.name",
        "team.name",
        "shots",
        "non_penalty_shots",
        "non_penalty_shots_in_box",
        "goals",
        "non_penalty_goals",
        "xg",
        "npxg",
        "average_npxg_per_shot",
        "xa",
    ]

    print(
        attacking[
            columns
        ]
        .sort_values(
            [
                "npxg",
                "xa",
            ],
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()