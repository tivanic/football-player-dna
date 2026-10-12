from player_dna.data.statsbomb import load_events
from player_dna.features.defending import (
    build_defending_features,
)


MATCH_ID = 3895074


def main() -> None:
    events = load_events(MATCH_ID)

    features = build_defending_features(events)

    columns = [
        "player.name",
        "team.name",
        "pressures",
        "ball_recoveries",
        "successful_recoveries",
        "offensive_recoveries",
        "tackles",
        "successful_tackles",
        "tackle_success_pct",
        "aerial_duels_lost",
        "interceptions",
        "successful_interceptions",
        "blocks",
        "dribbled_past",
        "fouls_committed",
    ]

    features = features.sort_values(
        "pressures",
        ascending=False,
    )

    print(
        features[columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()