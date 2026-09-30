from player_dna.data.statsbomb import load_matches


COMPETITION_ID = 9
SEASON_ID = 281


def main() -> None:
    matches = load_matches(COMPETITION_ID, SEASON_ID)

    columns = [
        "match_id",
        "match_date",
        "home_team.home_team_name",
        "away_team.away_team_name",
        "home_score",
        "away_score",
    ]

    print(f"Number of matches: {len(matches)}")
    print()
    print(matches[columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()