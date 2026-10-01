from pathlib import Path

import pandas as pd

from player_dna.data.statsbomb import load_events, load_matches
from player_dna.features.competition_profile import (
    aggregate_player_profiles,
)
from player_dna.features.player_profile import build_player_profile



COMPETITION_ID = 9
SEASON_ID = 281

OUTPUT_PATH = Path(
    "data/processed/bundesliga_2023_2024_player_profiles.parquet"
)


def main() -> None:
    matches = load_matches(
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
    )

    match_profiles = []

    total_matches = len(matches)

    for number, match in enumerate(
        matches.itertuples(),
        start=1,
    ):
        match_id = int(match.match_id)

        print(
            f"[{number}/{total_matches}] "
            f"Processing match {match_id}..."
        )

        events = load_events(match_id)

        profile = build_player_profile(events)

        profile["match_id"] = match_id

        match_profiles.append(profile)

    all_match_profiles = pd.concat(
        match_profiles,
        ignore_index=True,
    )

    player_profiles = aggregate_player_profiles(
        all_match_profiles
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    player_profiles.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved {len(player_profiles)} player profiles "
        f"to {OUTPUT_PATH}"
    )

    print()

    columns = [
        "player.name",
        "team.name",
        "primary_position",
        "position_group",
        "matches_played",
        "minutes_played",
        "pass_attempts_per90",
        "pass_completion_pct",
        "average_pass_length",
        "progressive_passes_per90",
        "final_third_entries_per90",
        "passes_into_box_per90",
    ]

    print(
        player_profiles[
            (
            player_profiles["minutes_played"] >= 450
            )
            & (
                player_profiles["position_group"] == "DM_CM"
            )
        ][columns]
        .sort_values(
            "minutes_played",
            ascending=False,
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()