from __future__ import annotations

from player_dna.data.statsbomb import (
    load_events,
)

from player_dna.features.spatial import (
    build_spatial_features,
)


MATCH_ID = 3754217


def main() -> None:
    events = load_events(
        MATCH_ID
    )

    spatial = build_spatial_features(
        events
    )

    names = [
        "Mesut Özil",
        "Francesc Fàbregas i Soler",
        "Eden Hazard",
        "Diego da Silva Costa",
        "Kurt Happy Zouma",
        "Nemanja Matić",
    ]

    columns = [
        "player.name",
        "team.name",

        "on_ball_actions",

        "average_on_ball_x",
        "average_on_ball_y",
        "average_on_ball_width",

        "defensive_third_action_share",
        "middle_third_action_share",
        "attacking_third_action_share",
        "wide_action_share",

        "average_receipt_x",
        "average_receipt_y",
        "average_receipt_width",

        "average_defensive_x",
        "average_defensive_y",
        "average_defensive_width",
    ]

    result = spatial[
        spatial[
            "player.name"
        ].isin(
            names
        )
    ]

    print(
        "SPATIAL FEATURES"
    )
    print(
        "----------------"
    )

    print(
        result[
            columns
        ]
        .sort_values(
            "average_on_ball_x",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()