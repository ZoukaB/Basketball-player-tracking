"""Interactive NBA half-court shot chart."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib.colors import load_colors
from lib.data import (
    default_shots_path,
    filter_shots,
    list_mock_shot_files,
    load_shots,
    normalize_shots,
    shot_metrics,
)
from lib.shot_chart import build_shot_chart

st.set_page_config(
    page_title="Shot Chart",
    layout="wide",
)

colors = load_colors()
st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {colors.background}; }}
        h1, h2, h3, p, label {{ color: {colors.text}; font-family: {colors.font}; }}
        [data-testid="stMetricValue"] {{ color: {colors.text}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Shot Chart")
st.caption("NBA half-court · 47 × 50 ft · SkillCorner colors")


def _load_from_sidebar() -> pd.DataFrame:
    mock_files = list_mock_shot_files()
    default = default_shots_path()
    labels = [p.name for p in mock_files]
    default_index = 0
    if default is not None and default.name in labels:
        default_index = labels.index(default.name)

    uploaded = st.file_uploader("Upload a shots CSV", type=["csv"])
    selected_name = None
    if labels:
        selected_name = st.selectbox(
            "Or use mock data",
            options=labels,
            index=default_index,
        )

    if uploaded is not None:
        return normalize_shots(pd.read_csv(uploaded))
    if selected_name:
        path = next(p for p in mock_files if p.name == selected_name)
        return load_shots(path)
    st.info("Add a shots CSV under outputs/mock_data or upload one.")
    return pd.DataFrame()


with st.sidebar:
    st.header("Filters")
    shots = _load_from_sidebar()

    if shots.empty:
        teams: list[str] = []
        players: list[str] = []
        shot_types: list[str] = []
        selected_teams: list[str] = []
        selected_players: list[str] = []
        selected_types: list[str] = []
        selected_results = ["Made", "Missed"]
        min_confidence = 0.0
    else:
        teams = sorted(shots["team"].dropna().unique().tolist())
        shot_types = sorted(shots["shot_type"].dropna().unique().tolist())
        selected_teams = st.multiselect("Team", teams, default=teams)
        player_pool = shots if not selected_teams else shots[shots["team"].isin(selected_teams)]
        players = sorted(player_pool["name"].dropna().unique().tolist())
        selected_players = st.multiselect("Player", players, default=players)
        selected_types = st.multiselect("Shot type", shot_types, default=shot_types)
        selected_results = st.multiselect(
            "Result",
            ["Made", "Missed"],
            default=["Made", "Missed"],
        )
        if shots["confidence"].notna().any():
            conf_min = float(shots["confidence"].min())
            conf_max = float(shots["confidence"].max())
            min_confidence = st.slider(
                "Min confidence",
                min_value=0.0,
                max_value=1.0,
                value=round(conf_min, 2),
                step=0.01,
            )
            if conf_max <= conf_min:
                min_confidence = conf_min
        else:
            min_confidence = None

if shots.empty:
    st.stop()

if not selected_teams or not selected_players or not selected_types or not selected_results:
    filtered = shots.iloc[0:0].copy()
else:
    filtered = filter_shots(
        shots,
        teams=selected_teams,
        players=selected_players,
        shot_types=selected_types,
        results=selected_results,
        min_confidence=min_confidence,
    )

metrics = shot_metrics(filtered)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("FGM", metrics["fgm"])
c2.metric("FGA", metrics["fga"])
c3.metric("FG%", f"{metrics['fg_pct'] * 100:.1f}")
c4.metric("3PM", metrics["tpm"])
c5.metric("3PA", metrics["tpa"])
c6.metric("3P%", f"{metrics['tp_pct'] * 100:.1f}")

fig = build_shot_chart(
    filtered,
    colors=colors,
    title=f"{len(filtered)} shots",
)
st.plotly_chart(fig, width="stretch")

display_cols = [
    "name",
    "team",
    "shot_type",
    "result",
    "x_court",
    "y_court",
    "distance",
    "confidence",
]
st.dataframe(filtered[display_cols], width="stretch", hide_index=True)
