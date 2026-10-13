from __future__ import annotations

import pandas as pd

from player_dna.data.statsbomb import (
    load_events,
    load_matches,
)


COMPETITION_ID = 2
SEASON_ID = 27

# Inspect several matches instead of only one.
# Optional event fields may not appear in every match.
N_MATCHES = 5


def existing_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    """
    Return only columns that actually exist
    in the DataFrame.
    """
    return [
        column
        for column in columns
        if column in df.columns
    ]


def main() -> None:
    matches = load_matches(
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
    )

    selected_matches = matches.head(
        N_MATCHES
    )

    print(
        f"Inspecting "
        f"{len(selected_matches)} "
        f"Premier League matches..."
    )

    all_events = []

    for match in selected_matches.itertuples():
        match_id = int(
            match.match_id
        )

        print(
            f"Loading match {match_id}..."
        )

        events = load_events(
            match_id
        )

        events["match_id"] = (
            match_id
        )

        all_events.append(
            events
        )

    events = pd.concat(
        all_events,
        ignore_index=True,
        sort=False,
    )

    print()
    print(
        f"Total events: "
        f"{len(events)}"
    )

    # --------------------------------------------------
    # SHOTS
    # --------------------------------------------------

    shots = events[
        events["type.name"]
        == "Shot"
    ].copy()

    print()
    print("=" * 80)
    print("SHOT EVENTS")
    print("=" * 80)

    print(
        f"Shots: "
        f"{len(shots)}"
    )

    shot_columns = sorted(
        [
            column
            for column in events.columns
            if column.startswith(
                "shot."
            )
        ]
    )

    print()
    print("SHOT COLUMNS")
    print("------------")

    for column in shot_columns:
        print(column)

    candidate_shot_columns = [
        "match_id",
        "id",
        "minute",
        "second",
        "player.id",
        "player.name",
        "team.name",
        "location",
        "shot.statsbomb_xg",
        "shot.outcome.name",
        "shot.type.name",
        "shot.body_part.name",
        "shot.end_location",
        "shot.key_pass_id",
        "shot.first_time",
        "shot.one_on_one",
        "shot.open_goal",
        "shot.deflected",
        "under_pressure",
    ]

    sample_columns = existing_columns(
        shots,
        candidate_shot_columns,
    )

    print()
    print("SHOT SAMPLE")
    print("-----------")

    print(
        shots[
            sample_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # SHOT OUTCOME
    # --------------------------------------------------

    if (
        "shot.outcome.name"
        in shots.columns
    ):
        print()
        print("SHOT OUTCOMES")
        print("-------------")

        print(
            shots[
                "shot.outcome.name"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # --------------------------------------------------
    # SHOT TYPE
    # --------------------------------------------------

    if (
        "shot.type.name"
        in shots.columns
    ):
        print()
        print("SHOT TYPES")
        print("----------")

        print(
            shots[
                "shot.type.name"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # --------------------------------------------------
    # BODY PART
    # --------------------------------------------------

    if (
        "shot.body_part.name"
        in shots.columns
    ):
        print()
        print("SHOT BODY PARTS")
        print("---------------")

        print(
            shots[
                "shot.body_part.name"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # --------------------------------------------------
    # XG
    # --------------------------------------------------

    if (
        "shot.statsbomb_xg"
        in shots.columns
    ):
        print()
        print("XG SUMMARY")
        print("----------")

        print(
            shots[
                "shot.statsbomb_xg"
            ]
            .describe()
            .to_string()
        )

    # --------------------------------------------------
    # PASSES THAT CREATE SHOTS
    # --------------------------------------------------

    passes = events[
        events["type.name"]
        == "Pass"
    ].copy()

    pass_columns = sorted(
        [
            column
            for column in events.columns
            if column.startswith(
                "pass."
            )
        ]
    )

    print()
    print("=" * 80)
    print("PASS / CHANCE CREATION")
    print("=" * 80)

    print()
    print("RELEVANT PASS COLUMNS")
    print("---------------------")

    relevant_pass_columns = [
        column
        for column in pass_columns
        if (
            "shot" in column.lower()
            or "assist" in column.lower()
            or "goal" in column.lower()
            or "key" in column.lower()
        )
    ]

    if relevant_pass_columns:
        for column in relevant_pass_columns:
            print(column)
    else:
        print(
            "No obvious shot/assist "
            "pass columns found."
        )

    candidate_pass_columns = [
        "match_id",
        "id",
        "minute",
        "second",
        "player.id",
        "player.name",
        "team.name",
        "pass.recipient.name",
        "pass.assisted_shot_id",
        "pass.shot_assist",
        "pass.goal_assist",
        "pass.through_ball",
        "pass.cross",
    ]

    existing_pass_columns = (
        existing_columns(
            passes,
            candidate_pass_columns,
        )
    )

    assisted_mask = pd.Series(
        False,
        index=passes.index,
    )

    if (
        "pass.assisted_shot_id"
        in passes.columns
    ):
        assisted_mask = (
            assisted_mask
            | passes[
                "pass.assisted_shot_id"
            ].notna()
        )

    if (
        "pass.shot_assist"
        in passes.columns
    ):
        assisted_mask = (
            assisted_mask
            | passes[
                "pass.shot_assist"
            ].fillna(False)
        )

    if (
        "pass.goal_assist"
        in passes.columns
    ):
        assisted_mask = (
            assisted_mask
            | passes[
                "pass.goal_assist"
            ].fillna(False)
        )

    assisted_passes = passes[
        assisted_mask
    ].copy()

    print()
    print(
        f"Passes linked to shots: "
        f"{len(assisted_passes)}"
    )

    if not assisted_passes.empty:
        print()
        print("ASSISTED PASS SAMPLE")
        print("--------------------")

        print(
            assisted_passes[
                existing_pass_columns
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # TEST PASS -> SHOT LINK
    # --------------------------------------------------

    if (
        "pass.assisted_shot_id"
        in passes.columns
        and "id" in shots.columns
    ):
        linked_passes = passes[
            passes[
                "pass.assisted_shot_id"
            ].notna()
        ].copy()

        shot_lookup_columns = (
            existing_columns(
                shots,
                [
                    "id",
                    "player.name",
                    "shot.statsbomb_xg",
                    "shot.outcome.name",
                    "shot.type.name",
                ],
            )
        )

        shot_lookup = shots[
            shot_lookup_columns
        ].copy()

        rename_columns = {
            "id":
                "shot_id",
            "player.name":
                "shooter_name",
            "shot.statsbomb_xg":
                "assisted_shot_xg",
            "shot.outcome.name":
                "assisted_shot_outcome",
            "shot.type.name":
                "assisted_shot_type",
        }

        shot_lookup = (
            shot_lookup.rename(
                columns=rename_columns
            )
        )

        linked = linked_passes.merge(
            shot_lookup,
            left_on="pass.assisted_shot_id",
            right_on="shot_id",
            how="left",
        )

        link_columns = (
            existing_columns(
                linked,
                [
                    "match_id",
                    "player.name",
                    "pass.recipient.name",
                    "pass.assisted_shot_id",
                    "shooter_name",
                    "assisted_shot_xg",
                    "assisted_shot_outcome",
                    "assisted_shot_type",
                ],
            )
        )

        print()
        print("=" * 80)
        print("PASS -> SHOT LINK TEST")
        print("=" * 80)

        print(
            linked[
                link_columns
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        if "shot_id" in linked.columns:
            matched = (
                linked["shot_id"]
                .notna()
                .sum()
            )

            print()
            print(
                f"Linked passes: "
                f"{len(linked)}"
            )

            print(
                f"Successfully matched shots: "
                f"{matched}"
            )


if __name__ == "__main__":
    main()