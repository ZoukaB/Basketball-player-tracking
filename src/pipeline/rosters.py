"""Default Knicks / Celtics rosters from the original notebook.

K-means team IDs (0/1) are arbitrary. After a test run, swap
``DEFAULT_TEAM_NAMES`` if names land on the wrong team.
"""

from __future__ import annotations

TEAM_ROSTERS: dict[str, dict[str, str]] = {
    "New York Knicks": {
        "55": "Hukporti",
        "1": "Payne",
        "0": "Wright",
        "11": "Brunson",
        "3": "Hart",
        "32": "Towns",
        "44": "Shamet",
        "25": "Bridges",
        "2": "McBride",
        "23": "Robinson",
        "8": "Anunoby",
        "4": "Dadiet",
        "5": "Achiuwa",
        "13": "Kolek",
    },
    "Boston Celtics": {
        "42": "Horford",
        "55": "Scheierman",
        "9": "White",
        "20": "Davison",
        "7": "Brown",
        "0": "Tatum",
        "27": "Walsh",
        "4": "Holiday",
        "8": "Porzingis",
        "40": "Kornet",
        "88": "Queta",
        "11": "Pritchard",
        "30": "Hauser",
        "12": "Craig",
        "26": "Tillman",
    },
}

# Cluster 0/1 assignment used in the notebook after inspecting jersey crops.
DEFAULT_TEAM_NAMES: dict[int, str] = {
    0: "Boston Celtics",
    1: "New York Knicks",
}

TEAM_COLORS: dict[str, str] = {
    "New York Knicks": "#006BB6",
    "Boston Celtics": "#007A33",
}
