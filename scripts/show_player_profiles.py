from player_dna.data.statsbomb import load_events
from player_dna.features.player_profile import build_player_profile


MATCH_ID = 3895074


def main() -> None:
    events = load_events(MATCH_ID)

    profile = build_player_profile(events)

    columns = [
        "player.name",
        "team.name",
        "minutes_played",
        "pass_attempts",
        "pass_attempts_per90",
        "pass_completion_pct",
        "average_pass_length",
        "progressive_passes",
        "progressive_passes_per90",
        "final_third_entries_per90",
        "passes_into_box_per90",
        "under_pressure_passes_per90",
        "shot_assists_per90",
    ]

    profile = profile[
        profile["minutes_played"] >= 10
    ]

    profile = profile.sort_values(
        by="progressive_passes_per90",
        ascending=False,
    )

    print(profile[columns].to_string(index=False))


if __name__ == "__main__":
    main()