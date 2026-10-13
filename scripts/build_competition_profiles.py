from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from player_dna.data.statsbomb import (
    load_events,
    load_matches,
)

from player_dna.features.competition_profile import (
    aggregate_player_profiles,
)

from player_dna.features.player_profile import (
    build_player_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build match-level and competition-level "
            "Player DNA profiles from StatsBomb Open Data."
        )
    )

    parser.add_argument(
        "--competition-id",
        type=int,
        required=True,
        help="StatsBomb competition ID.",
    )

    parser.add_argument(
        "--season-id",
        type=int,
        required=True,
        help="StatsBomb season ID.",
    )

    parser.add_argument(
        "--slug",
        type=str,
        required=True,
        help=(
            "Output name, for example "
            "'premier_league_2015_2016'."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    player_output_path = Path(
        f"data/processed/"
        f"{args.slug}_player_profiles.parquet"
    )

    match_output_path = Path(
        f"data/processed/"
        f"{args.slug}_match_profiles.parquet"
    )

    print(
        f"Loading competition "
        f"{args.competition_id}, "
        f"season {args.season_id}..."
    )

    matches = load_matches(
        competition_id=args.competition_id,
        season_id=args.season_id,
    )

    print(
        f"Matches found: {len(matches)}"
    )

    match_profiles = []

    total_matches = len(matches)

    for number, match in enumerate(
        matches.itertuples(),
        start=1,
    ):
        match_id = int(
            match.match_id
        )

        print(
            f"[{number}/{total_matches}] "
            f"Processing match {match_id}..."
        )

        events = load_events(
            match_id
        )

        profile = build_player_profile(
            events
        )

        profile["match_id"] = (
            match_id
        )

        profile["match_date"] = getattr(
            match,
            "match_date",
            None,
        )

        match_profiles.append(
            profile
        )

    all_match_profiles = pd.concat(
        match_profiles,
        ignore_index=True,
    )

    match_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_match_profiles.to_parquet(
        match_output_path,
        index=False,
    )

    print()
    print(
        f"Saved "
        f"{len(all_match_profiles)} "
        f"match-level player profiles to:"
    )
    print(
        match_output_path
    )

    player_profiles = (
        aggregate_player_profiles(
            all_match_profiles
        )
    )

    player_profiles.to_parquet(
        player_output_path,
        index=False,
    )

    print()
    print(
        f"Saved "
        f"{len(player_profiles)} "
        f"competition player profiles to:"
    )
    print(
        player_output_path
    )

    print()
    print(
        "POSITION GROUPS"
    )
    print(
        "---------------"
    )

    print(
        player_profiles[
            "position_group"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        "PLAYERS WITH 900+ MINUTES"
    )
    print(
        "-------------------------"
    )

    established = (
        player_profiles[
            player_profiles[
                "minutes_played"
            ] >= 900
        ]
        .sort_values(
            "minutes_played",
            ascending=False,
        )
    )

    columns = [
        "player.name",
        "team.name",
        "primary_position",
        "position_group",
        "matches_played",
        "minutes_played",
    ]

    print(
        established[
            columns
        ]
        .head(50)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()