from __future__ import annotations

from player_dna.data.statsbomb import (
    load_events,
)

from player_dna.features.player_profile import (
    build_player_profile,
)


MATCH_ID = 3754217


def main() -> None:
    events = load_events(
        MATCH_ID
    )

    profiles = build_player_profile(
        events
    )

    names = [
        "Francesc Fàbregas i Soler",
        "Pedro Eliezer Rodríguez Ledesma",
        "Kurt Happy Zouma",
        "Mesut Özil",
    ]

    columns = [
        "player.name",
        "team.name",
        "primary_position",
        "minutes_played",

        "shots",
        "non_penalty_shots",
        "non_penalty_shots_in_box",

        "npxg",
        "npxg_per90",

        "xa",
        "xa_per90",

        "average_npxg_per_shot",

        "final_third_receipts",
        "final_third_receipts_per90",
        "box_receipts",
        "box_receipts_per90",
        "receipts_under_pressure",
        "receipts_under_pressure_per90",

        "average_on_ball_x",
        "average_on_ball_width",
        "attacking_third_action_share",
        "wide_action_share",

        "average_receipt_x",
        "average_receipt_width",

        "average_defensive_x",
        "average_defensive_width",
    ]

    result = profiles[
        profiles[
            "player.name"
        ].isin(
            names
        )
    ]

    print(
        result[
            columns
        ]
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()