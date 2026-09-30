from __future__ import annotations

import pandas as pd
import requests


COMPETITIONS_URL = (
    "https://raw.githubusercontent.com/"
    "statsbomb/open-data/master/data/competitions.json"
)

MATCHES_URL = (
    "https://raw.githubusercontent.com/"
    "statsbomb/open-data/master/data/matches/{competition_id}/{season_id}.json"
)



def load_competitions() -> pd.DataFrame:
    """Load the list of competitions available in StatsBomb Open Data."""

    response = requests.get(COMPETITIONS_URL, timeout=30)
    response.raise_for_status()

    return pd.DataFrame(response.json())



def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Load matches for a competition and season."""

    url = MATCHES_URL.format(
        competition_id=competition_id,
        season_id=season_id,
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return pd.json_normalize(response.json())

    