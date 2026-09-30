from player_dna.data.statsbomb import load_events
from player_dna.features.player_match_stats import build_player_match_stats


MATCH_ID = 3895074


def main() -> None:
    events = load_events(MATCH_ID)

    stats = build_player_match_stats(events)

    stats = stats.sort_values(
        by="Pass",
        ascending=False,
    )

    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()