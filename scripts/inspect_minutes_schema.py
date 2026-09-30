from player_dna.data.statsbomb import load_events


MATCH_ID = 3895074


def main() -> None:
    events = load_events(MATCH_ID)

    print("MATCH END")
    print("---------")
    print(
        events[
            ["period", "minute", "second", "timestamp", "type.name"]
        ]
        .tail(20)
        .to_string(index=False)
    )

    print("\nSTARTING XI")
    print("-----------")

    starting_xi = events[events["type.name"] == "Starting XI"]

    for _, event in starting_xi.iterrows():
        print(f"\nTEAM: {event['team.name']}")

        lineup = event.get("tactics.lineup")

        if isinstance(lineup, list):
            for player in lineup:
                print(player)

    print("\nSUBSTITUTIONS")
    print("-------------")

    substitutions = events[events["type.name"] == "Substitution"]

    wanted_columns = [
        column
        for column in substitutions.columns
        if (
            column in {
                "minute",
                "second",
                "team.name",
                "player.id",
                "player.name",
            }
            or column.startswith("substitution.")
        )
    ]

    print(substitutions[wanted_columns].to_string(index=False))

    print("\nPLAYER ON / OFF")
    print("---------------")

    player_changes = events[
        events["type.name"].isin(["Player On", "Player Off"])
    ]

    print(
        player_changes[
            [
                "minute",
                "second",
                "type.name",
                "team.name",
                "player.id",
                "player.name",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()