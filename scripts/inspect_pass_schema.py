from player_dna.data.statsbomb import load_events


MATCH_ID = 3895074


def main() -> None:
    events = load_events(MATCH_ID)

    passes = events[events["type.name"] == "Pass"].copy()

    pass_columns = [
        column
        for column in passes.columns
        if column.startswith("pass.")
    ]

    print("PASS COLUMNS")
    print("------------")

    for column in pass_columns:
        print(column)

    print("\nPASS OUTCOMES")
    print("-------------")

    if "pass.outcome.name" in passes.columns:
        print(
            passes["pass.outcome.name"]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nPASS TYPES")
    print("----------")

    if "pass.type.name" in passes.columns:
        print(
            passes["pass.type.name"]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nPASS HEIGHT")
    print("-----------")

    if "pass.height.name" in passes.columns:
        print(
            passes["pass.height.name"]
            .value_counts(dropna=False)
            .to_string()
        )


if __name__ == "__main__":
    main()