from player_dna.data.statsbomb import load_events


MATCH_ID = 3895074


def main() -> None:
    events = load_events(MATCH_ID)

    print(f"Number of events: {len(events)}")
    print(f"Number of columns: {len(events.columns)}")

    print("\nEVENT TYPES")
    print(events["type.name"].value_counts().to_string())

    columns = [
        "minute",
        "second",
        "type.name",
        "team.name",
        "player.name",
        "location",
    ]

    print("\nFIRST 30 EVENTS")
    print(events[columns].head(30).to_string(index=False))

    passes = events[events["type.name"] == "Pass"]

    print("\nONE PASS EVENT")
    print()

    first_pass = passes.iloc[0]

    for column, value in first_pass.items():
        if value is not None and str(value) != "nan":
            print(f"{column}: {value}")


if __name__ == "__main__":
    main()