from player_dna.data.statsbomb import load_competitions


def main() -> None:
    competitions = load_competitions()

    columns = [
        "competition_id",
        "season_id",
        "country_name",
        "competition_name",
        "season_name",
    ]

    print(competitions[columns].to_string(index=False))


if __name__ == "__main__":
    main()