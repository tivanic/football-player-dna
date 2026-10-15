from __future__ import annotations

from html import escape
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from player_dna.model.player_dna_v1 import (
    MODEL_FEATURES,
    MODEL_NAMES,
    find_similar_players_v1,
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    "data/processed/"
    "premier_league_2015_2016_"
    "player_profiles.parquet"
)

MIN_MINUTES = 900


st.set_page_config(
    page_title="Player DNA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROLE_CATEGORY_ORDER = {
    "CB": [
        "defensive_involvement",
        "distribution",
        "progression",
        "receiving",
        "carrying",
        "creation",
        "goal_threat",
    ],

    "FB": [
        "progression",
        "carrying",
        "defensive_involvement",
        "creation",
        "distribution",
        "receiving",
        "goal_threat",
    ],

    "DM_CM": [
        "distribution",
        "progression",
        "defensive_involvement",
        "receiving",
        "carrying",
        "creation",
        "goal_threat",
    ],

    "AM_W": [
        "creation",
        "carrying",
        "progression",
        "receiving",
        "goal_threat",
        "distribution",
        "defensive_involvement",
    ],

    "FW": [
        "goal_threat",
        "receiving",
        "creation",
        "carrying",
        "progression",
        "defensive_involvement",
        "distribution",
    ],
}

# ============================================================
# HELPERS
# ============================================================

def render_html(html: str) -> None:
    """
    Render HTML directly instead of passing it
    through Streamlit's Markdown parser.
    """

    cleaned_html = dedent(
        html
    ).strip()

    if hasattr(
        st,
        "html",
    ):
        st.html(
            cleaned_html
        )

    else:
        # Fallback for older Streamlit versions.
        # Removing line breaks prevents Markdown
        # from interpreting nested HTML as code.
        st.markdown(
            cleaned_html.replace(
                "\n",
                "",
            ),
            unsafe_allow_html=True,
        )


# ============================================================
# STYLING
# ============================================================

render_html(
    """
    <style>
    :root {
        --bg: #f5f4ef;
        --surface: #ffffff;
        --text: #182019;
        --muted: #6f756f;
        --border: #d8ddd8;

        --green: #245f46;
        --green-soft: #e0e9e3;

        --brown: #8c6842;
        --bluegrey: #607480;

        --track: #e1e5e1;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    html,
    body,
    [class*="css"] {
        font-family:
            "Segoe UI",
            Inter,
            Arial,
            sans-serif;
    }

    h1,
    h2,
    h3,
    h4 {
        color: var(--text);
        letter-spacing: -0.025em;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ------------------------------------------
       GLOBAL TYPOGRAPHY
    ------------------------------------------ */

    .dna-kicker {
        color: var(--green);
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .dna-title {
        color: var(--text);
        font-size: 2.65rem;
        line-height: 1;
        font-weight: 780;
        letter-spacing: -0.055em;
        margin: 0;
    }

    .dna-subtitle {
        color: var(--muted);
        font-size: 0.96rem;
        line-height: 1.5;
        margin-top: 0.65rem;
    }

    .quiet-rule {
        height: 1px;
        background: var(--border);
        margin: 1.9rem 0;
    }

    .section-kicker {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.32rem;
    }

    .section-title {
        color: var(--text);
        font-size: 1.48rem;
        font-weight: 750;
        letter-spacing: -0.035em;
        margin-bottom: 0.4rem;
    }

    .section-note {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.55;
        margin-bottom: 1.3rem;
    }

    /* ------------------------------------------
       PLAYER HEADER
    ------------------------------------------ */

    .player-name {
        color: var(--text);
        font-size: 2.15rem;
        font-weight: 780;
        letter-spacing: -0.05em;
        line-height: 1.1;
    }

    .player-meta {
        color: var(--muted);
        font-size: 0.94rem;
        margin-top: 0.55rem;
        line-height: 1.5;
    }

    .model-note {
        color: var(--muted);
        font-size: 0.77rem;
        margin-top: 0.45rem;
    }

    /* ------------------------------------------
       EXPLANATION STRIP
    ------------------------------------------ */

    .info-strip {
        border-left: 3px solid var(--green);
        padding: 0.72rem 0 0.72rem 1rem;
        color: var(--muted);
        font-size: 0.87rem;
        line-height: 1.55;
        margin-top: 1.3rem;
    }

    /* ------------------------------------------
       DNA BARS
    ------------------------------------------ */

    .dna-bars {
        width: 100%;
        margin-top: 0.4rem;
    }

    .dna-row {
        margin-bottom: 1.12rem;
    }

    .dna-row-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.4rem;
    }

    .dna-row-label {
        color: var(--text);
        font-size: 0.91rem;
        font-weight: 680;
    }

    .dna-row-value {
        color: var(--muted);
        font-size: 0.8rem;
        text-align: right;
        white-space: nowrap;
    }

    .dna-track {
        width: 100%;
        height: 7px;
        border-radius: 999px;
        background: var(--track);
        overflow: hidden;
    }

    .dna-fill {
        height: 100%;
        border-radius: 999px;
        background: var(--green);
    }
    .dna-primary .dna-row-label {
        font-weight: 760;
    }

    .dna-primary .dna-track {
        height: 9px;
    }

    .dna-row:not(.dna-primary) {
        opacity: 0.72;
    }

    /* ------------------------------------------
       KEY METRICS
    ------------------------------------------ */

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        border-left: 1px solid var(--border);
        border-top: 1px solid var(--border);
        background: var(--surface);
    }

    .metric-cell {
        padding: 1rem 1.05rem;
        border-right: 1px solid var(--border);
        border-bottom: 1px solid var(--border);
        min-width: 0;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.7rem;
        line-height: 1.3;
        font-weight: 750;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.3rem;
        font-weight: 750;
        margin-top: 0.42rem;
    }

    .metric-percentile {
        color: var(--green);
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .role-metric-heading {
        margin-top: 1.35rem;
        margin-bottom: 0.55rem;

        color: var(--text);

        font-size: 0.82rem;
        font-weight: 760;

        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .role-metric-heading:first-of-type {
        margin-top: 0.2rem;
    }

    .role-metrics-grid {
        margin-bottom: 1.25rem;
    }

    /* ------------------------------------------
       SIMILAR PLAYERS
    ------------------------------------------ */

    .similar-list {
        border-top: 1px solid var(--border);
    }

    .similar-row {
        display: grid;
        grid-template-columns:
            42px
            minmax(0, 1.55fr)
            minmax(0, 1fr)
            minmax(150px, auto);

        align-items: center;
        gap: 1rem;

        padding: 0.92rem 0;
        border-bottom: 1px solid var(--border);
    }

    .similar-rank {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 750;
    }

    .similar-player {
        color: var(--text);
        font-size: 0.94rem;
        font-weight: 700;
    }

    .similar-position {
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: 0.16rem;
    }

    .similar-club {
        color: var(--muted);
        font-size: 0.84rem;
    }

    .similar-strength {
        text-align: right;
    }

    .similar-strength-main {
        color: var(--green);
        font-size: 0.82rem;
        font-weight: 700;
    }

    .similar-strength-sub {
        color: var(--muted);
        font-size: 0.72rem;
        margin-top: 0.14rem;
    }

    /* ------------------------------------------
       COMPARISON
    ------------------------------------------ */

    .compare-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--border);
    }

    .compare-label {
        color: var(--text);
        font-size: 0.88rem;
    }

    .compare-value {
        color: var(--muted);
        font-size: 0.82rem;
        white-space: nowrap;
    }

    .compare-small-heading {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-size: 0.7rem;
        font-weight: 750;
        margin-top: 1.2rem;
        margin-bottom: 0.25rem;
    }

    /* ------------------------------------------
       MOBILE
    ------------------------------------------ */

    @media (max-width: 760px) {

        .dna-title {
            font-size: 2.1rem;
        }

        .player-name {
            font-size: 1.8rem;
        }

        .metrics-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .similar-row {
            grid-template-columns:
                32px
                minmax(0, 1fr);
        }

        .similar-club {
            grid-column: 2;
        }

        .similar-strength {
            grid-column: 2;
            text-align: left;
        }
    }
    </style>
    """
)


# ============================================================
# TRANSLATIONS
# ============================================================

TEXT = {
    "HRV": {
        "competition": "Premier League 2015/16",
        "tagline": "Analiza stila igre iz StatsBomb event podataka",

        "choose_player": "Odaberi igrača",

        "style_explanation": (
            "Player DNA opisuje način igre, a ne ukupnu kvalitetu igrača. "
            "Svaki rezultat uspoređuje igrača isključivo s igračima njegove "
            "pozicijske grupe. Zato visoka obrambena uključenost napadača "
            "ne znači da je bolji branič od stopera."
        ),

        "playing_style": "Profil stila igre",
        "playing_style_note": (
            "Svaka vrijednost je percentil unutar iste uloge. "
            "Primjerice, 90. percentil znači da je ta karakteristika "
            "izraženija nego kod 90% usporedivih igrača."
        ),

        "compared_with": "Usporedba s",
        "same_role_players": "igrača iste uloge",

        "spatial_profile": "Prostorni profil",
        "spatial_note": (
            "Prosječne lokacije zabilježenih akcija. "
            "To nije sustav praćenja fizičke pozicije igrača."
        ),

        "on_ball": "S loptom",
        "receiving_point": "Primanje",
        "defensive_point": "Obrambene akcije",

        "own_goal": "Vlastiti gol",
        "opponent_goal": "Protivnički gol",

        "key_metrics": "Ključne metrike",
        "key_metrics_note": (
            "Stvarna vrijednost metrike i njen položaj u odnosu "
            "na druge igrače iste uloge."
        ),
        #!
        "role_profile": "Profil uloge",
        "role_profile_note": (
            "Istaknute metrike prilagođene su ulozi igrača. "
            "Percentili i dalje opisuju koliko je neka karakteristika izražena "
            "u odnosu na igrače iste pozicijske grupe."
        ),

        "defensive_activity": "Obrambena aktivnost",
        "on_ball_profile": "Igra s loptom",

        "progression_width": "Progresija i igra po širini",
        "defensive_work": "Obrambeni rad",

        "midfield_control": "Kontrola sredine",
        "progression_creation": "Progresija i kreacija",

        "creation_progression": "Kreacija i progresija",
        "threat_receiving": "Prijetnja i primanje lopte",

        "attacking_threat": "Napadačka prijetnja",
        "link_pressing": "Povezivanje igre i presing",
        #!

        "similar_players": "Najsličniji igrači",
        "similar_note": (
            "Model traži najsličniji stil igre među igračima "
            "iste široke pozicijske grupe."
        ),

        "very_strong": "Vrlo snažno podudaranje",
        "strong": "Snažno podudaranje",
        "good": "Dobro podudaranje",
        "moderate": "Umjereno podudaranje",

        "top_pair": "među najboljih {pct}% parova",

        "compare_with": "Usporedi s",
        "comparison": "Usporedba profila",

        "most_similar_in": "Najsličniji su u",
        "biggest_differences": "Najveće razlike",

        "advanced": "Napredne metrike",
        "metric": "Metrika",

        "model": "Model",
        "features": "featurea",

        "pct": "{value}. percentil",
    },

    "ENG": {
        "competition": "Premier League 2015/16",
        "tagline": "Playing-style analysis from StatsBomb event data",

        "choose_player": "Choose a player",

        "style_explanation": (
            "Player DNA describes playing style, not overall player quality. "
            "Every score compares the player only with players in the same "
            "role group. Therefore a forward with high defensive involvement "
            "is not being rated as a better defender than a centre-back."
        ),

        "playing_style": "Playing-style profile",
        "playing_style_note": (
            "Each value is a percentile within the same role. "
            "For example, the 90th percentile means the characteristic "
            "is more pronounced than for 90% of comparable players."
        ),

        "compared_with": "Compared with",
        "same_role_players": "players in the same role",

        "spatial_profile": "Spatial profile",
        "spatial_note": (
            "Average locations of recorded actions. "
            "This is not continuous player tracking."
        ),

        "on_ball": "On-ball",
        "receiving_point": "Receiving",
        "defensive_point": "Defensive actions",

        "own_goal": "Own goal",
        "opponent_goal": "Opponent goal",

        "key_metrics": "Key metrics",
        "key_metrics_note": (
            "Actual metric values together with their position "
            "relative to other players in the same role."
        ),
        #!
        "role_profile": "Role profile",
        "role_profile_note": (
            "Highlighted metrics are tailored to the player's role. "
            "Percentiles still describe how pronounced each characteristic is "
            "relative to players in the same position group."
        ),

        "defensive_activity": "Defensive activity",
        "on_ball_profile": "On-ball play",

        "progression_width": "Progression & width",
        "defensive_work": "Defensive work",

        "midfield_control": "Midfield control",
        "progression_creation": "Progression & creation",

        "creation_progression": "Creation & progression",
        "threat_receiving": "Threat & receiving",

        "attacking_threat": "Attacking threat",
        "link_pressing": "Link play & pressing",
        #!

        "similar_players": "Closest style matches",
        "similar_note": (
            "The model searches for the closest playing styles "
            "within the same broad position group."
        ),

        "very_strong": "Very strong match",
        "strong": "Strong match",
        "good": "Good match",
        "moderate": "Moderate match",

        "top_pair": "top {pct}% of role pairs",

        "compare_with": "Compare with",
        "comparison": "Profile comparison",

        "most_similar_in": "Most alike in",
        "biggest_differences": "Biggest differences",

        "advanced": "Advanced metrics",
        "metric": "Metric",

        "model": "Model",
        "features": "features",

        "pct": "{value}th percentile",
    },
}


CATEGORY_LABELS = {
    "HRV": {
        "distribution": "Distribucija lopte",
        "progression": "Progresija",
        "carrying": "Nošenje lopte",
        "creation": "Kreacija prilika",
        "goal_threat": "Prijetnja golu",
        "receiving": "Primanje lopte",
        "defensive_involvement":
            "Obrambena uključenost",
    },

    "ENG": {
        "distribution": "Distribution",
        "progression": "Progression",
        "carrying": "Ball carrying",
        "creation": "Chance creation",
        "goal_threat": "Goal threat",
        "receiving": "Receiving",
        "defensive_involvement":
            "Defensive involvement",
    },
}


ROLE_LABELS = {
    "HRV": {
        "CB": "Stoper",
        "FB": "Bočni branič",
        "DM_CM":
            "Centralni / defenzivni vezni",
        "AM_W":
            "Ofenzivni vezni / krilo",
        "FW": "Napadač",
    },

    "ENG": {
        "CB": "Centre-back",
        "FB": "Full-back",
        "DM_CM":
            "Central / defensive midfield",
        "AM_W":
            "Attacking midfield / wing",
        "FW": "Forward",
    },
}


POSITION_HR = {
    "Goalkeeper":
        "Vratar",

    "Right Center Back":
        "Desni stoper",

    "Center Back":
        "Stoper",

    "Left Center Back":
        "Lijevi stoper",

    "Right Back":
        "Desni bek",

    "Left Back":
        "Lijevi bek",

    "Right Wing Back":
        "Desni wing-back",

    "Left Wing Back":
        "Lijevi wing-back",

    "Right Defensive Midfield":
        "Desni defenzivni vezni",

    "Center Defensive Midfield":
        "Defenzivni vezni",

    "Left Defensive Midfield":
        "Lijevi defenzivni vezni",

    "Right Center Midfield":
        "Desni centralni vezni",

    "Center Midfield":
        "Centralni vezni",

    "Left Center Midfield":
        "Lijevi centralni vezni",

    "Right Attacking Midfield":
        "Desni ofenzivni vezni",

    "Center Attacking Midfield":
        "Ofenzivni vezni",

    "Left Attacking Midfield":
        "Lijevi ofenzivni vezni",

    "Right Midfield":
        "Desni vezni",

    "Left Midfield":
        "Lijevi vezni",

    "Right Wing":
        "Desno krilo",

    "Left Wing":
        "Lijevo krilo",

    "Right Forward":
        "Desni napadač",

    "Center Forward":
        "Centralni napadač",

    "Left Forward":
        "Lijevi napadač",

    "Right Center Forward":
        "Desni centralni napadač",

    "Left Center Forward":
        "Lijevi centralni napadač",

    "Secondary Striker":
        "Polušpica",
}


def tx(
    language: str,
    key: str,
) -> str:
    return TEXT[
        language
    ][
        key
    ]


def role_label(
    role: str,
    language: str,
) -> str:
    return (
        ROLE_LABELS[
            language
        ]
        .get(
            role,
            role,
        )
    )


def position_label(
    position: str,
    language: str,
) -> str:
    if language == "HRV":
        return POSITION_HR.get(
            position,
            position,
        )

    return position


# ============================================================
# DNA PRESENTATION CATEGORIES
# ============================================================

DNA_CATEGORIES = {
    "distribution": [
        "pass_attempts_per90",
        "pass_completion_pct",
        "average_pass_length",
        "forward_pass_pct",
        "switches_per90",
    ],

    "progression": [
        "progressive_passes_per90",
        "final_third_entries_per90",
        "passes_into_box_per90",
        "progressive_carries_per90",
        "carries_into_final_third_per90",
    ],

    "carrying": [
        "total_carry_distance_per90",
        "average_carry_length",
        "progressive_carries_per90",
        "dribbles_per90",
        "successful_dribbles_per90",
    ],

    "creation": [
        "xa_per90",
        "shot_assists_per90",
        "through_balls_per90",
        "passes_into_box_per90",
    ],

    "goal_threat": [
        "non_penalty_shots_per90",
        "non_penalty_shots_in_box_per90",
        "npxg_per90",
        "average_npxg_per_shot",
    ],

    "receiving": [
        "final_third_receipts_per90",
        "box_receipts_per90",
        "receipts_under_pressure_per90",
    ],

    "defensive_involvement": [
        "pressures_per90",
        "ball_recoveries_per90",
        "tackles_per90",
        "interceptions_per90",
        "blocks_per90",
    ],
}


# ============================================================
# KEY METRICS BY ROLE
# ============================================================

ROLE_PROFILE_SECTIONS = {
    "CB": [
        {
            "title_key": "defensive_activity",
            "features": [
                "ball_recoveries_per90",
                "interceptions_per90",
                "blocks_per90",
                "tackles_per90",
                "average_defensive_x",
            ],
        },
        {
            "title_key": "on_ball_profile",
            "features": [
                "pass_completion_pct",
                "progressive_passes_per90",
                "under_pressure_passes_per90",
                "average_pass_length",
            ],
        },
    ],

    "FB": [
        {
            "title_key": "progression_width",
            "features": [
                "progressive_passes_per90",
                "progressive_carries_per90",
                "crosses_per90",
                "xa_per90",
                "final_third_receipts_per90",
            ],
        },
        {
            "title_key": "defensive_work",
            "features": [
                "pressures_per90",
                "ball_recoveries_per90",
                "tackles_per90",
                "interceptions_per90",
                "average_defensive_x",
            ],
        },
    ],

    "DM_CM": [
        {
            "title_key": "midfield_control",
            "features": [
                "pass_attempts_per90",
                "pass_completion_pct",
                "ball_recoveries_per90",
                "pressures_per90",
                "average_on_ball_x",
            ],
        },
        {
            "title_key": "progression_creation",
            "features": [
                "progressive_passes_per90",
                "passes_into_box_per90",
                "xa_per90",
                "progressive_carries_per90",
                "receipts_under_pressure_per90",
            ],
        },
    ],

    "AM_W": [
        {
            "title_key": "creation_progression",
            "features": [
                "xa_per90",
                "shot_assists_per90",
                "passes_into_box_per90",
                "progressive_passes_per90",
                "progressive_carries_per90",
            ],
        },
        {
            "title_key": "threat_receiving",
            "features": [
                "npxg_per90",
                "non_penalty_shots_per90",
                "box_receipts_per90",
                "final_third_receipts_per90",
                "dribbles_per90",
            ],
        },
    ],

    "FW": [
        {
            "title_key": "attacking_threat",
            "features": [
                "npxg_per90",
                "non_penalty_shots_per90",
                "non_penalty_shots_in_box_per90",
                "average_npxg_per_shot",
                "box_receipts_per90",
            ],
        },
        {
            "title_key": "link_pressing",
            "features": [
                "xa_per90",
                "progressive_passes_per90",
                "receipts_under_pressure_per90",
                "pressures_per90",
                "pass_attempts_per90",
            ],
        },
    ],
}


# ============================================================
# FEATURE LABELS
# ============================================================

FEATURE_LABELS_HR = {
    "pass_attempts_per90":
        "Dodavanja /90",

    "pass_completion_pct":
        "Točnost dodavanja",

    "average_pass_length":
        "Prosječna duljina dodavanja",

    "forward_pass_pct":
        "Dodavanja prema naprijed",

    "progressive_passes_per90":
        "Progresivna dodavanja /90",

    "final_third_entries_per90":
        "Ulazi u zadnju trećinu /90",

    "passes_into_box_per90":
        "Dodavanja u kazneni prostor /90",

    "under_pressure_passes_per90":
        "Dodavanja pod pritiskom /90",

    "crosses_per90":
        "Centaršutevi /90",

    "switches_per90":
        "Promjene strane /90",

    "through_balls_per90":
        "Proigravanja kroz liniju /90",

    "shot_assists_per90":
        "Dodavanja za udarac /90",

    "total_carry_distance_per90":
        "Udaljenost nošenja lopte /90",

    "average_carry_length":
        "Prosječna duljina nošenja",

    "progressive_carries_per90":
        "Progresivna nošenja /90",

    "carries_into_final_third_per90":
        "Nošenja u zadnju trećinu /90",

    "carries_into_box_per90":
        "Nošenja u kazneni prostor /90",

    "dribbles_per90":
        "Driblinzi /90",

    "successful_dribbles_per90":
        "Uspješni driblinzi /90",

    "ball_receipts_per90":
        "Primanja lopte /90",

    "incomplete_ball_receipts_per90":
        "Neuspjela primanja /90",

    "dispossessed_per90":
        "Izgubljene lopte duelom /90",

    "miscontrols_per90":
        "Loše kontrole /90",

    "pressures_per90":
        "Pritisci /90",

    "ball_recoveries_per90":
        "Osvojene lopte /90",

    "successful_recoveries_per90":
        "Uspješna osvajanja lopte /90",

    "offensive_recoveries_per90":
        "Visoka osvajanja lopte /90",

    "tackles_per90":
        "Startovi /90",

    "successful_tackles_per90":
        "Uspješni startovi /90",

    "aerial_duels_lost_per90":
        "Izgubljeni zračni dueli /90",

    "interceptions_per90":
        "Presječena dodavanja /90",

    "successful_interceptions_per90":
        "Uspješna presijecanja /90",

    "blocks_per90":
        "Blokovi /90",

    "dribbled_past_per90":
        "Pretrpljeni driblinzi /90",

    "fouls_committed_per90":
        "Prekršaji /90",

    "non_penalty_shots_per90":
        "Udarci bez penala /90",

    "non_penalty_shots_in_box_per90":
        "Udarci iz kaznenog prostora /90",

    "npxg_per90":
        "npxG /90",

    "average_npxg_per_shot":
        "Prosječni npxG po udarcu",

    "xa_per90":
        "xA /90",

    "final_third_receipts_per90":
        "Primanja u zadnjoj trećini /90",

    "box_receipts_per90":
        "Primanja u kaznenom prostoru /90",

    "receipts_under_pressure_per90":
        "Primanja pod pritiskom /90",

    "average_on_ball_x":
        "Prosječna dubina s loptom",

    "average_on_ball_width":
        "Prosječna širina aktivnosti",

    "average_receipt_x":
        "Prosječna dubina primanja",

    "average_receipt_width":
        "Prosječna širina primanja",

    "average_defensive_x":
        "Dubina obrambenih akcija",

    "average_defensive_width":
        "Širina obrambenih akcija",

    "attacking_third_action_share":
        "Aktivnost u zadnjoj trećini",

    "wide_action_share":
        "Aktivnost u širokim zonama",
}


FEATURE_LABELS_EN = {
    "pass_attempts_per90":
        "Passes /90",

    "pass_completion_pct":
        "Pass completion",

    "average_pass_length":
        "Average pass length",

    "forward_pass_pct":
        "Forward passes",

    "progressive_passes_per90":
        "Progressive passes /90",

    "final_third_entries_per90":
        "Final-third entries /90",

    "passes_into_box_per90":
        "Passes into box /90",

    "under_pressure_passes_per90":
        "Passes under pressure /90",

    "crosses_per90":
        "Crosses /90",

    "switches_per90":
        "Switches /90",

    "through_balls_per90":
        "Through balls /90",

    "shot_assists_per90":
        "Shot assists /90",

    "total_carry_distance_per90":
        "Carry distance /90",

    "average_carry_length":
        "Average carry length",

    "progressive_carries_per90":
        "Progressive carries /90",

    "carries_into_final_third_per90":
        "Carries into final third /90",

    "carries_into_box_per90":
        "Carries into box /90",

    "dribbles_per90":
        "Dribbles /90",

    "successful_dribbles_per90":
        "Successful dribbles /90",

    "ball_receipts_per90":
        "Ball receipts /90",

    "incomplete_ball_receipts_per90":
        "Incomplete receipts /90",

    "dispossessed_per90":
        "Dispossessed /90",

    "miscontrols_per90":
        "Miscontrols /90",

    "pressures_per90":
        "Pressures /90",

    "ball_recoveries_per90":
        "Ball recoveries /90",

    "successful_recoveries_per90":
        "Successful recoveries /90",

    "offensive_recoveries_per90":
        "High recoveries /90",

    "tackles_per90":
        "Tackles /90",

    "successful_tackles_per90":
        "Successful tackles /90",

    "aerial_duels_lost_per90":
        "Aerial duels lost /90",

    "interceptions_per90":
        "Interceptions /90",

    "successful_interceptions_per90":
        "Successful interceptions /90",

    "blocks_per90":
        "Blocks /90",

    "dribbled_past_per90":
        "Dribbled past /90",

    "fouls_committed_per90":
        "Fouls committed /90",

    "non_penalty_shots_per90":
        "Non-penalty shots /90",

    "non_penalty_shots_in_box_per90":
        "Non-penalty box shots /90",

    "npxg_per90":
        "npxG /90",

    "average_npxg_per_shot":
        "Average npxG per shot",

    "xa_per90":
        "xA /90",

    "final_third_receipts_per90":
        "Final-third receipts /90",

    "box_receipts_per90":
        "Box receipts /90",

    "receipts_under_pressure_per90":
        "Receipts under pressure /90",

    "average_on_ball_x":
        "Average on-ball depth",

    "average_on_ball_width":
        "Average on-ball width",

    "average_receipt_x":
        "Average receiving depth",

    "average_receipt_width":
        "Average receiving width",

    "average_defensive_x":
        "Defensive action depth",

    "average_defensive_width":
        "Defensive action width",

    "attacking_third_action_share":
        "Attacking-third activity",

    "wide_action_share":
        "Wide-area activity",
}


def human_feature_name(
    feature: str,
    language: str,
) -> str:

    if language == "HRV":
        return FEATURE_LABELS_HR.get(
            feature,
            feature,
        )

    return FEATURE_LABELS_EN.get(
        feature,
        feature,
    )


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_profiles() -> pd.DataFrame:
    return pd.read_parquet(
        DATA_PATH
    )


def stable_player_key(
    row: pd.Series,
) -> str:

    player_id = int(
        row["player.id"]
    )

    return (
        f"{player_id}"
        f"::{row['team.name']}"
    )


def player_display_label(
    row: pd.Series,
) -> str:

    return (
        f"{row['player.name']} "
        f"— {row['team.name']}"
    )


def get_player_by_key(
    profiles: pd.DataFrame,
    player_key: str,
) -> pd.Series:

    player_id_text, team_name = (
        player_key.split(
            "::",
            1,
        )
    )

    player_id = int(
        player_id_text
    )

    result = profiles[
        (
            profiles[
                "player.id"
            ] == player_id
        )
        & (
            profiles[
                "team.name"
            ] == team_name
        )
    ]

    if result.empty:
        raise ValueError(
            "Player could not be found."
        )

    return result.iloc[0]


# ============================================================
# POPULATION / PERCENTILES
# ============================================================

def role_population(
    profiles: pd.DataFrame,
    role: str,
) -> pd.DataFrame:

    return profiles[
        (
            profiles[
                "position_group"
            ] == role
        )
        & (
            profiles[
                "minutes_played"
            ] >= MIN_MINUTES
        )
    ].copy()


def feature_percentile(
    population: pd.DataFrame,
    feature: str,
    value: float,
) -> float:

    if feature not in population.columns:
        return 50.0

    series = pd.to_numeric(
        population[
            feature
        ],
        errors="coerce",
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    valid = series.dropna()

    if (
        valid.empty
        or pd.isna(value)
    ):
        return 50.0

    percentile = (
        (valid <= value)
        .mean()
        * 100
    )

    return float(
        percentile
    )


def build_dna_scores(
    profiles: pd.DataFrame,
    player: pd.Series,
) -> dict[str, float]:

    role = player[
        "position_group"
    ]

    population = role_population(
        profiles,
        role,
    )

    scores = {}

    for (
        category,
        features,
    ) in DNA_CATEGORIES.items():

        values = []

        for feature in features:

            if feature not in profiles.columns:
                continue

            raw_value = pd.to_numeric(
                pd.Series(
                    [
                        player.get(
                            feature,
                            np.nan,
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if pd.isna(
                raw_value
            ):
                continue

            values.append(
                feature_percentile(
                    population,
                    feature,
                    float(
                        raw_value
                    ),
                )
            )

        if values:
            scores[
                category
            ] = float(
                np.mean(
                    values
                )
            )

        else:
            scores[
                category
            ] = 50.0

    return scores


# ============================================================
# SCORE WORDING
# ============================================================

def score_descriptor(
    score: float,
    language: str,
) -> str:

    if language == "HRV":

        if score >= 90:
            return "Vrlo izraženo"

        if score >= 75:
            return "Izraženo"

        if score >= 60:
            return "Iznad prosjeka"

        if score >= 40:
            return "Umjereno"

        if score >= 25:
            return "Slabije izraženo"

        return "Vrlo slabo izraženo"

    if score >= 90:
        return "Very pronounced"

    if score >= 75:
        return "Pronounced"

    if score >= 60:
        return "Above average"

    if score >= 40:
        return "Moderate"

    if score >= 25:
        return "Less pronounced"

    return "Very low"


def percentile_text(
    percentile: float,
    language: str,
) -> str:

    value = int(
        round(
            percentile
        )
    )

    if language == "HRV":
        return (
            f"{value}. percentil"
        )

    # English ordinal
    if 10 <= value % 100 <= 20:
        suffix = "th"

    else:
        suffix = {
            1: "st",
            2: "nd",
            3: "rd",
        }.get(
            value % 10,
            "th",
        )

    return (
        f"{value}{suffix} percentile"
    )


# ============================================================
# DNA BARS
# ============================================================

def render_dna_bars(
    scores: dict[str, float],
    language: str,
    role: str,
) -> None:

    category_order = (
        ROLE_CATEGORY_ORDER[
            role
        ]
    )

    rows = []

    for index, category in enumerate(
        category_order
    ):
        score = scores[
            category
        ]

        label = (
            CATEGORY_LABELS[
                language
            ][category]
        )

        description = score_descriptor(
            score,
            language,
        )

        pct_label = percentile_text(
            score,
            language,
        )

        primary_class = (
            " dna-primary"
            if index < 4
            else ""
        )

        rows.append(
            f"""
            <div class="dna-row{primary_class}">
                <div class="dna-row-header">
                    <div class="dna-row-label">
                        {escape(label)}
                    </div>

                    <div class="dna-row-value">
                        {escape(pct_label)}
                        ·
                        {escape(description)}
                    </div>
                </div>

                <div class="dna-track">
                    <div
                        class="dna-fill"
                        style="width:{score:.1f}%;">
                    </div>
                </div>
            </div>
            """
        )

    render_html(
        f"""
        <div class="dna-bars">
            {''.join(rows)}
        </div>
        """
    )

# ============================================================
# KEY METRICS
# ============================================================

def format_metric_value(
    feature: str,
    value: float,
) -> str:

    if pd.isna(
        value
    ):
        return "—"

    if (
        feature.endswith(
            "_pct"
        )
        or feature.endswith(
            "_share"
        )
    ):
        return (
            f"{value:.1f}%"
        )

    if feature in {
        "average_npxg_per_shot",
        "npxg_per90",
        "xa_per90",
    }:
        return (
            f"{value:.3f}"
        )

    if feature.endswith(
        "_per90"
    ):
        return (
            f"{value:.2f}"
        )

    if feature.startswith(
        "average_"
    ):
        return (
            f"{value:.1f}"
        )

    return (
        f"{value:.2f}"
    )

#
def render_role_profile(
    profiles: pd.DataFrame,
    player: pd.Series,
    language: str,
) -> None:

    role = player[
        "position_group"
    ]

    population = role_population(
        profiles,
        role,
    )

    sections = ROLE_PROFILE_SECTIONS[
        role
    ]

    for section in sections:

        section_title = tx(
            language,
            section[
                "title_key"
            ],
        )

        render_html(
            f"""
            <div class="role-metric-heading">
                {escape(section_title)}
            </div>
            """
        )

        cells = []

        for feature in section[
            "features"
        ]:

            value = pd.to_numeric(
                pd.Series(
                    [
                        player.get(
                            feature,
                            np.nan,
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            percentile = feature_percentile(
                population,
                feature,
                value,
            )

            cells.append(
                f"""
                <div class="metric-cell">

                    <div class="metric-label">
                        {escape(
                            human_feature_name(
                                feature,
                                language,
                            )
                        )}
                    </div>

                    <div class="metric-value">
                        {escape(
                            format_metric_value(
                                feature,
                                value,
                            )
                        )}
                    </div>

                    <div class="metric-percentile">
                        {escape(
                            percentile_text(
                                percentile,
                                language,
                            )
                        )}
                    </div>

                </div>
                """
            )

        render_html(
            f"""
            <div class="metrics-grid role-metrics-grid">
                {''.join(cells)}
            </div>
            """
        )
#

# ============================================================
# SIMILARITY CONTEXT
# ============================================================

@st.cache_data
def role_similarity_distribution(
    role: str,
) -> np.ndarray:

    profiles = load_profiles()

    population = role_population(
        profiles,
        role,
    )

    features = (
        MODEL_FEATURES[
            role
        ]
    )

    values = (
        population[
            features
        ]
        .astype(float)
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    medians = (
        values
        .median()
        .fillna(
            0.0
        )
    )

    values = values.fillna(
        medians
    )

    scaler = (
        StandardScaler()
    )

    scaled = (
        scaler.fit_transform(
            values
        )
    )

    similarity_matrix = (
        cosine_similarity(
            scaled,
            scaled,
        )
    )

    upper_triangle = (
        np.triu_indices(
            len(
                population
            ),
            k=1,
        )
    )

    distribution = (
        similarity_matrix[
            upper_triangle
        ]
    )

    return np.sort(
        distribution
    )


def similarity_percentile(
    role: str,
    score: float,
) -> float:

    distribution = (
        role_similarity_distribution(
            role
        )
    )

    if len(
        distribution
    ) == 0:
        return 50.0

    index = (
        np.searchsorted(
            distribution,
            score,
            side="right",
        )
    )

    return float(
        index
        / len(
            distribution
        )
        * 100
    )


def similarity_wording(
    percentile: float,
    language: str,
) -> str:

    if percentile >= 97:
        return tx(
            language,
            "very_strong",
        )

    if percentile >= 92:
        return tx(
            language,
            "strong",
        )

    if percentile >= 82:
        return tx(
            language,
            "good",
        )

    return tx(
        language,
        "moderate",
    )


def render_similar_players(
    similar: pd.DataFrame,
    role: str,
    language: str,
) -> None:

    rows = []

    for _, row in (
        similar.iterrows()
    ):

        percentile = (
            similarity_percentile(
                role,
                float(
                    row[
                        "similarity_score"
                    ]
                ),
            )
        )

        top_percentage = max(
            1,
            int(
                np.ceil(
                    100
                    - percentile
                )
            ),
        )

        strength = (
            similarity_wording(
                percentile,
                language,
            )
        )

        top_text = (
            tx(
                language,
                "top_pair",
            )
            .format(
                pct=top_percentage
            )
        )

        position = (
            position_label(
                str(
                    row[
                        "primary_position"
                    ]
                ),
                language,
            )
        )

        rows.append(
            f"""
            <div class="similar-row">

                <div class="similar-rank">
                    {int(row["rank"]):02d}
                </div>

                <div>
                    <div class="similar-player">
                        {escape(
                            str(
                                row[
                                    "player.name"
                                ]
                            )
                        )}
                    </div>

                    <div class="similar-position">
                        {escape(
                            position
                        )}
                    </div>
                </div>

                <div class="similar-club">
                    {escape(
                        str(
                            row[
                                "team.name"
                            ]
                        )
                    )}
                </div>

                <div class="similar-strength">

                    <div class="similar-strength-main">
                        {escape(
                            strength
                        )}
                    </div>

                    <div class="similar-strength-sub">
                        {escape(
                            top_text
                        )}
                    </div>

                </div>

            </div>
            """
        )

    render_html(
        f"""
        <div class="similar-list">
            {''.join(rows)}
        </div>
        """
    )


# ============================================================
# SPATIAL PITCH
# ============================================================

def build_spatial_pitch(
    player: pd.Series,
    language: str,
) -> go.Figure:

    pitch_line = "#aeb7b0"
    pitch_bg = "#eef1ed"

    traces = [
        {
            "label":
                tx(
                    language,
                    "defensive_point",
                ),

            "x":
                "average_defensive_x",

            "y":
                "average_defensive_y",

            "symbol":
                "triangle-up",

            "size":
                14,

            "color":
                "#607480",
        },

        {
            "label":
                tx(
                    language,
                    "on_ball",
                ),

            "x":
                "average_on_ball_x",

            "y":
                "average_on_ball_y",

            "symbol":
                "circle",

            "size":
                16,

            "color":
                "#245f46",
        },

        {
            "label":
                tx(
                    language,
                    "receiving_point",
                ),

            "x":
                "average_receipt_x",

            "y":
                "average_receipt_y",

            "symbol":
                "diamond",

            "size":
                13,

            "color":
                "#8c6842",
        },
    ]

    fig = go.Figure()

    # Pitch
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=120,
        y1=80,
        line={
            "color":
                pitch_line,
            "width":
                1.5,
        },
    )

    # Halfway line
    fig.add_shape(
        type="line",
        x0=60,
        y0=0,
        x1=60,
        y1=80,
        line={
            "color":
                pitch_line,
            "width":
                1,
        },
    )

    # Centre circle
    fig.add_shape(
        type="circle",
        x0=50,
        x1=70,
        y0=30,
        y1=50,
        line={
            "color":
                pitch_line,
            "width":
                1,
        },
    )

    # Centre point
    fig.add_trace(
        go.Scatter(
            x=[60],
            y=[40],
            mode="markers",
            marker={
                "size": 3,
                "color":
                    pitch_line,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Penalty areas
    fig.add_shape(
        type="rect",
        x0=0,
        y0=18,
        x1=18,
        y1=62,
        line={
            "color":
                pitch_line,
            "width":
                1,
        },
    )

    fig.add_shape(
        type="rect",
        x0=102,
        y0=18,
        x1=120,
        y1=62,
        line={
            "color":
                pitch_line,
            "width":
                1,
        },
    )

    # Six-yard boxes
    fig.add_shape(
        type="rect",
        x0=0,
        y0=30,
        x1=6,
        y1=50,
        line={
            "color":
                pitch_line,
            "width":
                1,
        },
    )

    fig.add_shape(
        type="rect",
        x0=114,
        y0=30,
        x1=120,
        y1=50,
        line={
            "color":
                pitch_line,
            "width":
                1,
        },
    )

    # Player points
    for trace in traces:

        x_value = player.get(
            trace[
                "x"
            ],
            np.nan,
        )

        y_value = player.get(
            trace[
                "y"
            ],
            np.nan,
        )

        if (
            pd.isna(
                x_value
            )
            or pd.isna(
                y_value
            )
        ):
            continue

        fig.add_trace(
            go.Scatter(
                x=[
                    x_value
                ],
                y=[
                    y_value
                ],

                mode="markers",

                name=trace[
                    "label"
                ],

                marker={
                    "symbol":
                        trace[
                            "symbol"
                        ],

                    "size":
                        trace[
                            "size"
                        ],

                    "color":
                        trace[
                            "color"
                        ],

                    "line": {
                        "color":
                            "#ffffff",
                        "width":
                            1.4,
                    },
                },

                hovertemplate=(
                    f"<b>{trace['label']}</b>"
                    f"<br>x: {x_value:.1f}"
                    f"<br>y: {y_value:.1f}"
                    "<extra></extra>"
                ),
            )
        )

    # Goal-direction annotations
    fig.add_annotation(
        x=2,
        y=84,
        text=(
            "← "
            + tx(
                language,
                "own_goal",
            )
        ),
        showarrow=False,
        xanchor="left",
        font={
            "size": 10,
            "color":
                "#6f756f",
        },
    )

    fig.add_annotation(
        x=118,
        y=84,
        text=(
            tx(
                language,
                "opponent_goal",
            )
            + " →"
        ),
        showarrow=False,
        xanchor="right",
        font={
            "size": 10,
            "color":
                "#6f756f",
        },
    )

    fig.update_xaxes(
        range=[
            -2,
            122,
        ],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        fixedrange=True,
    )

    fig.update_yaxes(
        range=[
            88,
            -3,
        ],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True,
    )

    fig.update_layout(
        height=410,

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        plot_bgcolor=(
            pitch_bg
        ),

        margin={
            "l": 8,
            "r": 8,
            "t": 10,
            "b": 55,
        },

        legend={
            "orientation":
                "h",

            "x":
                0.5,

            "xanchor":
                "center",

            "y":
                -0.08,

            "yanchor":
                "top",

            "font": {
                "size":
                    10,
            },
        },

        hoverlabel={
            "bgcolor":
                "#ffffff",

            "font_color":
                "#182019",
        },
    )

    return fig


# ============================================================
# COMPARISON RADAR
# ============================================================

def build_comparison_radar(
    first_scores: dict[str, float],
    first_name: str,
    second_scores: dict[str, float],
    second_name: str,
    language: str,
) -> go.Figure:

    keys = list(
        first_scores.keys()
    )

    labels = [
        CATEGORY_LABELS[
            language
        ][
            key
        ]
        for key in keys
    ]

    first_values = [
        first_scores[
            key
        ]
        for key in keys
    ]

    second_values = [
        second_scores[
            key
        ]
        for key in keys
    ]

    closed_labels = (
        labels
        + [
            labels[0]
        ]
    )

    closed_first = (
        first_values
        + [
            first_values[0]
        ]
    )

    closed_second = (
        second_values
        + [
            second_values[0]
        ]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=closed_first,
            theta=closed_labels,

            mode="lines",
            fill="toself",

            name=first_name,

            line={
                "color":
                    "#245f46",
                "width":
                    2,
            },

            fillcolor=(
                "rgba(36,95,70,0.13)"
            ),
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=closed_second,
            theta=closed_labels,

            mode="lines",
            fill="toself",

            name=second_name,

            line={
                "color":
                    "#8c6842",
                "width":
                    2,
            },

            fillcolor=(
                "rgba(140,104,66,0.10)"
            ),
        )
    )

    fig.update_layout(
        height=480,

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        font={
            "color":
                "#182019",
            "size":
                11,
        },

        polar={
            "bgcolor":
                "rgba(0,0,0,0)",

            "radialaxis": {
                "visible":
                    True,

                "range":
                    [
                        0,
                        100,
                    ],

                "tickvals":
                    [
                        25,
                        50,
                        75,
                        100,
                    ],

                "gridcolor":
                    "#d8ddd8",

                "linecolor":
                    "#d8ddd8",
            },

            "angularaxis": {
                "gridcolor":
                    "#d8ddd8",

                "linecolor":
                    "#d8ddd8",
            },
        },

        legend={
            "orientation":
                "h",

            "x":
                0.5,

            "xanchor":
                "center",

            "y":
                -0.13,
        },

        margin={
            "l":
                60,

            "r":
                60,

            "t":
                30,

            "b":
                75,
        },
    )

    return fig


# ============================================================
# CATEGORY COMPARISON
# ============================================================

def category_comparison(
    first_scores: dict[str, float],
    second_scores: dict[str, float],
) -> tuple[
    list[
        tuple[
            str,
            float,
        ]
    ],
    list[
        tuple[
            str,
            float,
        ]
    ],
]:

    differences = []

    for category in first_scores:

        difference = abs(
            first_scores[
                category
            ]
            - second_scores[
                category
            ]
        )

        differences.append(
            (
                category,
                difference,
            )
        )

    most_similar = sorted(
        differences,
        key=lambda item:
            item[1],
    )[:3]

    biggest_differences = (
        sorted(
            differences,
            key=lambda item:
                item[1],
            reverse=True,
        )[:3]
    )

    return (
        most_similar,
        biggest_differences,
    )


def render_category_comparison(
    items: list[
        tuple[
            str,
            float,
        ]
    ],
    language: str,
) -> None:

    rows = []

    for (
        category,
        difference,
    ) in items:

        label = (
            CATEGORY_LABELS[
                language
            ][
                category
            ]
        )

        rows.append(
            f"""
            <div class="compare-row">

                <div class="compare-label">
                    {escape(
                        label
                    )}
                </div>

                <div class="compare-value">
                    Δ {difference:.0f}
                </div>

            </div>
            """
        )

    render_html(
        "".join(
            rows
        )
    )


# ============================================================
# APP DATA
# ============================================================

profiles = (
    load_profiles()
)


eligible = profiles[
    (
        profiles[
            "minutes_played"
        ]
        >= MIN_MINUTES
    )
    & (
        profiles[
            "position_group"
        ]
        .isin(
            MODEL_FEATURES.keys()
        )
    )
].copy()


eligible = (
    eligible
    .sort_values(
        "player.name"
    )
    .reset_index(
        drop=True
    )
)


eligible[
    "_player_key"
] = eligible.apply(
    stable_player_key,
    axis=1,
)


eligible[
    "_player_label"
] = eligible.apply(
    player_display_label,
    axis=1,
)


PLAYER_LABELS = dict(
    zip(
        eligible[
            "_player_key"
        ],
        eligible[
            "_player_label"
        ],
    )
)


PLAYER_KEYS = (
    eligible[
        "_player_key"
    ]
    .tolist()
)


# ============================================================
# SESSION STATE
# ============================================================

# if (
#     "language"
#     not in st.session_state
# ):
#     st.session_state[
#         "language"
#     ] = "HRV"
if (
    "language"
    not in st.session_state
):
    st.session_state[
        "language"
    ] = "HRV"


if (
    "language_selector"
    not in st.session_state
):
    st.session_state[
        "language_selector"
    ] = st.session_state[
        "language"
    ]
##
def update_language() -> None:
    st.session_state[
        "language"
    ] = st.session_state[
        "language_selector"
    ]
##

if (
    "selected_player_key"
    not in st.session_state
    or st.session_state[
        "selected_player_key"
    ]
    not in PLAYER_KEYS
):

    # Default to Özil only the FIRST time.
    ozil = eligible[
        eligible[
            "player.name"
        ]
        .str.contains(
            "Özil",
            case=False,
            regex=False,
        )
    ]

    if not ozil.empty:
        default_player_key = (
            ozil.iloc[0][
                "_player_key"
            ]
        )

    else:
        default_player_key = (
            PLAYER_KEYS[0]
        )

    st.session_state[
        "selected_player_key"
    ] = default_player_key


# ============================================================
# HEADER
# ============================================================

header_left, header_right = (
    st.columns(
        [
            5,
            1,
        ]
    )
)


with header_left:

    render_html(
        """
        <div class="dna-kicker">
            Football analytics
        </div>

        <div class="dna-title">
            Player DNA
        </div>
        """
    )


with header_right:

    st.radio(
        "Language",
        options=[
            "HRV",
            "ENG",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="language_selector",
        on_change=update_language,
    )


language = st.session_state[
    "language"
]

render_html(
    f"""
    <div class="dna-subtitle">
        {escape(
            tx(
                language,
                "competition",
            )
        )}
        &nbsp;·&nbsp;
        {escape(
            tx(
                language,
                "tagline",
            )
        )}
    </div>
    """
)


# ============================================================
# PLAYER SELECTOR
# ============================================================

selected_player_key = (
    st.selectbox(
        tx(
            language,
            "choose_player",
        ),

        options=PLAYER_KEYS,

        format_func=lambda key:
            PLAYER_LABELS[
                key
            ],

        key="selected_player_key",
    )
)


selected_row = (
    get_player_by_key(
        eligible,
        selected_player_key,
    )
)


target, similar = (
    find_similar_players_v1(
        profiles,

        player_name=(
            selected_row[
                "player.name"
            ]
        ),

        team_name=(
            selected_row[
                "team.name"
            ]
        ),

        min_minutes=(
            MIN_MINUTES
        ),

        top_n=5,
    )
)


role = target[
    "position_group"
]


population = (
    role_population(
        profiles,
        role,
    )
)


# ============================================================
# PLAYER HEADER
# ============================================================

render_html(
    '<div class="quiet-rule"></div>'
)


render_html(
    f"""
    <div class="player-name">
        {escape(
            str(
                target[
                    "player.name"
                ]
            )
        )}
    </div>

    <div class="player-meta">

        {escape(
            str(
                target[
                    "team.name"
                ]
            )
        )}

        &nbsp;·&nbsp;

        {escape(
            position_label(
                str(
                    target[
                        "primary_position"
                    ]
                ),
                language,
            )
        )}

        &nbsp;·&nbsp;

        {escape(
            role_label(
                role,
                language,
            )
        )}

        &nbsp;·&nbsp;

        {target["minutes_played"]:.0f} min

    </div>
    """
)


render_html(
    f"""
    <div class="info-strip">
        {escape(
            tx(
                language,
                "style_explanation",
            )
        )}
    </div>
    """
)


# ============================================================
# DNA + SPATIAL PROFILE
# ============================================================

target_scores = (
    build_dna_scores(
        profiles,
        target,
    )
)


left_col, right_col = (
    st.columns(
        [
            1,
            1,
        ],
        gap="large",
    )
)


with left_col:

    render_html(
        f"""
        <div class="section-kicker">
            Player DNA
        </div>

        <div class="section-title">
            {escape(
                tx(
                    language,
                    "playing_style",
                )
            )}
        </div>

        <div class="section-note">

            {escape(
                tx(
                    language,
                    "playing_style_note",
                )
            )}

            <br>

            {escape(
                tx(
                    language,
                    "compared_with",
                )
            )}

            <strong>
                {len(population)}
            </strong>

            {escape(
                tx(
                    language,
                    "same_role_players",
                )
            )}.

        </div>
        """
    )

    render_dna_bars(
        target_scores,
        language,
        role,
    )
    # render_dna_bars(
    #     target_scores,
    #     language,
    # )


with right_col:

    render_html(
        f"""
        <div class="section-kicker">
            Spatial DNA
        </div>

        <div class="section-title">
            {escape(
                tx(
                    language,
                    "spatial_profile",
                )
            )}
        </div>

        <div class="section-note">
            {escape(
                tx(
                    language,
                    "spatial_note",
                )
            )}
        </div>
        """
    )


    st.plotly_chart(
        build_spatial_pitch(
            target,
            language,
        ),
        use_container_width=True,
        config={
            "displayModeBar":
                False,
        },
    )


# ============================================================
# KEY METRICS
# ============================================================

render_html(
    '<div class="quiet-rule"></div>'
)


render_html(
    f"""
    <div class="section-kicker">
        Role context
    </div>

    <div class="section-title">
        {escape(
            tx(
                language,
                "role_profile",
            )
        )}
    </div>

    <div class="section-note">
        {escape(
            tx(
                language,
                "role_profile_note",
            )
        )}
    </div>
    """
)


render_role_profile(
    profiles,
    target,
    language,
)


# ============================================================
# SIMILAR PLAYERS
# ============================================================

render_html(
    '<div class="quiet-rule"></div>'
)


render_html(
    f"""
    <div class="section-kicker">
        Similarity
    </div>

    <div class="section-title">
        {escape(
            tx(
                language,
                "similar_players",
            )
        )}
    </div>

    <div class="section-note">
        {escape(
            tx(
                language,
                "similar_note",
            )
        )}
    </div>
    """
)


render_similar_players(
    similar,
    role,
    language,
)


# ============================================================
# COMPARISON SELECTOR
# ============================================================

comparison_options = {}


for _, row in similar.iterrows():

    comparison_key = (
        f"{int(row['player.id'])}"
        f"::{row['team.name']}"
    )

    comparison_label = (
        f"{int(row['rank']):02d}"
        f" · "
        f"{row['player.name']}"
        f" — "
        f"{row['team.name']}"
    )

    comparison_options[
        comparison_key
    ] = comparison_label


comparison_keys = list(
    comparison_options.keys()
)


# Reset comparison only when TARGET player changes.
previous_target = (
    st.session_state.get(
        "_comparison_target"
    )
)


if (
    previous_target
    != selected_player_key
):

    st.session_state[
        "_comparison_target"
    ] = selected_player_key

    st.session_state[
        "selected_comparison_key"
    ] = comparison_keys[0]


if (
    "selected_comparison_key"
    not in st.session_state
    or st.session_state[
        "selected_comparison_key"
    ]
    not in comparison_keys
):

    st.session_state[
        "selected_comparison_key"
    ] = comparison_keys[0]


selected_comparison_key = (
    st.selectbox(
        tx(
            language,
            "compare_with",
        ),

        options=(
            comparison_keys
        ),

        format_func=lambda key:
            comparison_options[
                key
            ],

        key=(
            "selected_comparison_key"
        ),
    )
)


comparison = (
    get_player_by_key(
        profiles,
        selected_comparison_key,
    )
)


comparison_scores = (
    build_dna_scores(
        profiles,
        comparison,
    )
)


# ============================================================
# COMPARISON
# ============================================================

render_html(
    '<div class="quiet-rule"></div>'
)


render_html(
    f"""
    <div class="section-kicker">
        Head-to-head
    </div>

    <div class="section-title">
        {escape(
            str(
                target[
                    "player.name"
                ]
            )
        )}

        &nbsp;vs&nbsp;

        {escape(
            str(
                comparison[
                    "player.name"
                ]
            )
        )}
    </div>
    """
)


radar_col, explanation_col = (
    st.columns(
        [
            1.35,
            1,
        ],
        gap="large",
    )
)


with radar_col:

    st.plotly_chart(
        build_comparison_radar(
            target_scores,

            str(
                target[
                    "player.name"
                ]
            ),

            comparison_scores,

            str(
                comparison[
                    "player.name"
                ]
            ),

            language,
        ),

        use_container_width=True,

        config={
            "displayModeBar":
                False,
        },
    )


with explanation_col:

    (
        most_similar,
        biggest_differences,
    ) = category_comparison(
        target_scores,
        comparison_scores,
    )


    render_html(
        f"""
        <div class="compare-small-heading">
            {escape(
                tx(
                    language,
                    "most_similar_in",
                )
            )}
        </div>
        """
    )


    render_category_comparison(
        most_similar,
        language,
    )


    render_html(
        f"""
        <div class="compare-small-heading">
            {escape(
                tx(
                    language,
                    "biggest_differences",
                )
            )}
        </div>
        """
    )


    render_category_comparison(
        biggest_differences,
        language,
    )


# ============================================================
# ADVANCED METRICS
# ============================================================

with st.expander(
    tx(
        language,
        "advanced",
    )
):

    model_features = (
        MODEL_FEATURES[
            role
        ]
    )

    rows = []

    for feature in model_features:

        target_value = pd.to_numeric(
            pd.Series(
                [
                    target.get(
                        feature,
                        np.nan,
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]


        comparison_value = pd.to_numeric(
            pd.Series(
                [
                    comparison.get(
                        feature,
                        np.nan,
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]


        target_pct = (
            feature_percentile(
                population,
                feature,
                target_value,
            )
        )


        comparison_pct = (
            feature_percentile(
                population,
                feature,
                comparison_value,
            )
        )


        rows.append(
            {
                tx(
                    language,
                    "metric",
                ):
                    human_feature_name(
                        feature,
                        language,
                    ),

                str(
                    target[
                        "player.name"
                    ]
                ):
                    format_metric_value(
                        feature,
                        target_value,
                    ),

                (
                    str(
                        target[
                            "player.name"
                        ]
                    )
                    + " · pct"
                ):
                    percentile_text(
                        target_pct,
                        language,
                    ),

                str(
                    comparison[
                        "player.name"
                    ]
                ):
                    format_metric_value(
                        feature,
                        comparison_value,
                    ),

                (
                    str(
                        comparison[
                            "player.name"
                        ]
                    )
                    + " · pct"
                ):
                    percentile_text(
                        comparison_pct,
                        language,
                    ),
            }
        )


    advanced = pd.DataFrame(
        rows
    )


    st.dataframe(
        advanced,
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
        f"{tx(language, 'model')}: "
        f"{MODEL_NAMES[role]} · "
        f"{len(MODEL_FEATURES[role])} "
        f"{tx(language, 'features')}"
    )