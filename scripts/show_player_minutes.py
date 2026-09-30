from player_dna.data.statsbomb import load_events
from player_dna.features.minutes import build_player_minutes


MATCH_ID = 3895074


def main() -> None:
    events = load_events(MATCH_ID)

    minutes = build_player_minutes(events)

    print(minutes.to_string(index=False))


if __name__ == "__main__":
    main()