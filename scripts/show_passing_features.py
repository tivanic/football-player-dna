from player_dna.data.statsbomb import load_events
from player_dna.features.passing import build_passing_features


MATCH_ID = 3895074


def main() -> None:
    events = load_events(MATCH_ID)

    features = build_passing_features(events)

    features = features.sort_values(
        by="pass_attempts",
        ascending=False,
    )

    print(features.to_string(index=False))


if __name__ == "__main__":
    main()