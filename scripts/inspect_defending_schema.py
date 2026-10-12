from player_dna.data.statsbomb import load_events


MATCH_ID = 3895074


def inspect_event(
    events,
    event_type,
    prefixes,
):
    subset = events[
        events["type.name"] == event_type
    ]

    print(f"\n{event_type.upper()}")
    print("-" * len(event_type))

    print(f"Events: {len(subset)}")

    columns = [
        column
        for column in subset.columns
        if any(
            column.startswith(prefix)
            for prefix in prefixes
        )
    ]

    for column in columns:
        print(column)

    for column in columns:
        if column.endswith(".name"):
            print(f"\n{column}")

            print(
                subset[column]
                .value_counts(dropna=False)
                .to_string()
            )


def main() -> None:
    events = load_events(MATCH_ID)

    inspect_event(
        events,
        "Pressure",
        ["pressure."],
    )

    inspect_event(
        events,
        "Ball Recovery",
        ["ball_recovery."],
    )

    inspect_event(
        events,
        "Duel",
        ["duel."],
    )

    inspect_event(
        events,
        "Interception",
        ["interception."],
    )

    inspect_event(
        events,
        "Block",
        ["block."],
    )

    inspect_event(
        events,
        "Dribbled Past",
        ["dribbled_past."],
    )

    inspect_event(
        events,
        "Foul Committed",
        ["foul_committed."],
    )

    print("\nONE DUEL EVENT")
    print("--------------")

    duels = events[
        events["type.name"] == "Duel"
    ]

    if not duels.empty:
        event = duels.iloc[0]

        for column, value in event.items():
            if (
                value is not None
                and str(value) != "nan"
                and (
                    column in {
                        "minute",
                        "second",
                        "player.name",
                        "team.name",
                        "location",
                        "under_pressure",
                    }
                    or column.startswith("duel.")
                )
            ):
                print(f"{column}: {value}")

    print("\nONE BALL RECOVERY EVENT")
    print("-----------------------")

    recoveries = events[
        events["type.name"] == "Ball Recovery"
    ]

    if not recoveries.empty:
        event = recoveries.iloc[0]

        for column, value in event.items():
            if (
                value is not None
                and str(value) != "nan"
                and (
                    column in {
                        "minute",
                        "second",
                        "player.name",
                        "team.name",
                        "location",
                    }
                    or column.startswith(
                        "ball_recovery."
                    )
                )
            ):
                print(f"{column}: {value}")


if __name__ == "__main__":
    main()