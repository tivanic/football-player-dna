from player_dna.data.statsbomb import load_events


MATCH_ID = 3895074


def print_columns(events, event_type, prefixes):
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


def main() -> None:
    events = load_events(MATCH_ID)

    print_columns(
        events,
        "Carry",
        ["carry."],
    )

    print_columns(
        events,
        "Dribble",
        ["dribble."],
    )

    print_columns(
        events,
        "Ball Receipt*",
        ["ball_receipt."],
    )

    print_columns(
        events,
        "Dispossessed",
        ["dispossessed."],
    )

    print_columns(
        events,
        "Miscontrol",
        ["miscontrol."],
    )

    dribbles = events[
        events["type.name"] == "Dribble"
    ]

    if "dribble.outcome.name" in dribbles.columns:
        print("\nDRIBBLE OUTCOMES")
        print("----------------")

        print(
            dribbles["dribble.outcome.name"]
            .value_counts(dropna=False)
            .to_string()
        )

    receipts = events[
        events["type.name"] == "Ball Receipt*"
    ]

    if "ball_receipt.outcome.name" in receipts.columns:
        print("\nBALL RECEIPT OUTCOMES")
        print("---------------------")

        print(
            receipts["ball_receipt.outcome.name"]
            .value_counts(dropna=False)
            .to_string()
        )

    carries = events[
        events["type.name"] == "Carry"
    ]

    print("\nONE CARRY EVENT")
    print("---------------")

    if not carries.empty:
        carry = carries.iloc[0]

        for column, value in carry.items():
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
                    or column.startswith("carry.")
                )
            ):
                print(f"{column}: {value}")


if __name__ == "__main__":
    main()