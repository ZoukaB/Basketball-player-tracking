"""Shot chart dashboard."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from lib.colors import load_colors, theme_css
from lib.data import default_shots_path, filter_shots, load_shots, shot_metrics
from lib.shot_chart import build_shot_chart

ALL_PLAYERS = "All players"
ALL = "All"

st.set_page_config(
    page_title="Shot Chart",
    layout="wide",
    initial_sidebar_state="collapsed",
)

colors = load_colors()
st.markdown(theme_css(colors), unsafe_allow_html=True)


@st.cache_data
def _load_default_shots() -> pd.DataFrame:
    path = default_shots_path()
    if path is None:
        return pd.DataFrame()
    return load_shots(path)


def _placeholder_svg() -> str:
    return (
        '<div class="placeholder-bust">'
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="12" cy="8" r="4"/>'
        '<path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>'
        "</svg></div>"
    )


def _on_team_change() -> None:
    st.session_state.player_choice = ALL_PLAYERS


shots = _load_default_shots()
if shots.empty:
    st.error("No shots CSV found in outputs/mock_data.")
    st.stop()

teams = sorted(shots["team"].dropna().unique().tolist())
if "selected_team" not in st.session_state:
    st.session_state.selected_team = teams[0]
if st.session_state.selected_team not in teams:
    st.session_state.selected_team = teams[0]
if "player_choice" not in st.session_state:
    st.session_state.player_choice = ALL_PLAYERS

team_players = sorted(
    shots.loc[shots["team"] == st.session_state.selected_team, "name"]
    .dropna()
    .unique()
    .tolist()
)
player_options = [ALL_PLAYERS, *team_players]
if st.session_state.player_choice not in player_options:
    st.session_state.player_choice = ALL_PLAYERS

team_col, player_col, _ = st.columns([2, 2, 5], gap="small")
with team_col:
    with st.popover("Team", width="stretch"):
        st.radio(
            "Team",
            teams,
            key="selected_team",
            on_change=_on_team_change,
            label_visibility="collapsed",
        )
with player_col:
    with st.popover("Player", width="stretch"):
        st.radio(
            "Player",
            player_options,
            key="player_choice",
            label_visibility="collapsed",
        )

selected_team = st.session_state.selected_team
selected_player = (
    None
    if st.session_state.player_choice == ALL_PLAYERS
    else st.session_state.player_choice
)

shot_type_options = ["All", *sorted(shots["shot_type"].dropna().unique().tolist())]
if "shot_type_filter" not in st.session_state:
    st.session_state.shot_type_filter = ALL
if "result_filter" not in st.session_state:
    st.session_state.result_filter = ALL
if st.session_state.shot_type_filter not in shot_type_options:
    st.session_state.shot_type_filter = ALL

filtered = filter_shots(
    shots,
    team=selected_team,
    player=selected_player,
    shot_type=st.session_state.shot_type_filter,
    result=st.session_state.result_filter,
)
metrics = shot_metrics(filtered)
fg = f"{metrics['fg_pct'] * 100:.1f}%"
tp = f"{metrics['tp_pct'] * 100:.1f}%" if metrics["tpa"] else "—"

info_col, court_col = st.columns([1, 2], gap="small")

with info_col:
    if selected_player:
        player_rows = shots[
            (shots["team"] == selected_team) & (shots["name"] == selected_player)
        ]
        jersey = ""
        if len(player_rows) and player_rows["player_id"].notna().any():
            jersey = str(player_rows["player_id"].dropna().iloc[0])
        heading = selected_player
        team_line = selected_team
        jersey_line = f"#{jersey}" if jersey else None
    else:
        heading = selected_team
        team_line = None
        jersey_line = None

    details = []
    if selected_player:
        details.append(f"<div><strong>Team</strong> {team_line}</div>")
        if jersey_line:
            details.append(f"<div><strong>No.</strong> {jersey_line}</div>")
    details.append(f"<div><strong>Shots</strong> {metrics['fga']}</div>")
    details.append(f"<div><strong>FG%</strong> {fg}</div>")
    details.append(f"<div><strong>3P%</strong> {tp}</div>")

    st.markdown(
        f"""
        <div class="info-panel">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.35rem;">
            {_placeholder_svg()}
            <p class="info-name">{heading}</p>
          </div>
          <div class="info-meta">{"".join(details)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.selectbox("Shot type", shot_type_options, key="shot_type_filter")
    st.selectbox("Result", ["All", "Made", "Missed"], key="result_filter")

with court_col:
    fig = build_shot_chart(filtered, colors=colors, title=None, height=700)
    st.plotly_chart(
        fig,
        width="stretch",
        height="stretch",
        config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
    )
