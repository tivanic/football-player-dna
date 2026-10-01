from player_dna.data.statsbomb import load_events
from player_dna.features.possession import (
    build_possession_features,
)


MATCH_ID = 3895074


def main() -> None:
    events = load_events(MATCH_ID)

    features = build_possession_features(events)

    columns = [
        "player.name",
        "team.name",
        "carries",
        "total_carry_distance",
        "average_carry_length",
        "progressive_carries",
        "carries_into_final_third",
        "carries_into_box",
        "dribbles",
        "successful_dribbles",
        "successful_dribble_pct",
        "ball_receipts",
        "dispossessed",
        "miscontrols",
    ]

    features = features.sort_values(
        "carries",
        ascending=False,
    )

    print(
        features[columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()