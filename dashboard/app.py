"""Shot chart dashboard.

Reads outputs/<run>/dashboard_shots.csv, built from a collect_shots.py run by
dashboard/build_data.py. Filter by team and player; a selected player shows
their game stats (FG%, 3P%, 2P, layups).

Run (from repo root):
    .venv/Scripts/python.exe -m streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from lib.colors import load_colors, theme_css
from lib.data import (
    REPO_ROOT,
    clock_label,
    filter_shots,
    list_dashboard_files,
    load_shots,
    quarter_label,
    shot_metrics,
    time_label,
)
from lib.headshots import headshot_uri, logo_uri
from lib.shot_chart import build_shot_chart

ALL_TEAMS = "All teams"
ALL_PLAYERS = "All players"
ALL = "All"
QUARTER_S = 720  # 12:00
CLOCK_STEP_S = 10
# Game-clock slider runs like the real clock: 12:00 on the left, 0:00 on the right.
CLOCK_OPTIONS = [clock_label(s) for s in range(QUARTER_S, -1, -CLOCK_STEP_S)]
CLOCK_TO_S = {label: s for label, s in zip(CLOCK_OPTIONS, range(QUARTER_S, -1, -CLOCK_STEP_S))}
FULL_CLOCK = (CLOCK_OPTIONS[0], CLOCK_OPTIONS[-1])

st.set_page_config(
    page_title="Shot Chart",
    layout="wide",
    initial_sidebar_state="collapsed",
)

colors = load_colors()
st.markdown(theme_css(colors), unsafe_allow_html=True)


@st.cache_data
def _load(path: str, mtime: float) -> pd.DataFrame:
    # mtime in the key reloads the file when build_data.py rewrites it.
    return load_shots(path)


def _placeholder_svg() -> str:
    return (
        '<div class="placeholder-bust">'
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="12" cy="8" r="4"/>'
        '<path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>'
        "</svg></div>"
    )


def _pct(made: int, attempts: int) -> str:
    return f"{made / attempts * 100:.1f}%" if attempts else "—"


def _on_team_change() -> None:
    st.session_state.player_choice = ALL_PLAYERS


files = list_dashboard_files()
if not files:
    st.error("No outputs/*/dashboard_shots.csv found. Run collect_shots.py, then dashboard/build_data.py.")
    st.stop()
file_labels = {str(p.relative_to(REPO_ROOT)): p for p in files}

st.session_state.setdefault("data_file", next(iter(file_labels)))
if st.session_state.data_file not in file_labels:
    st.session_state.data_file = next(iter(file_labels))
data_path = file_labels[st.session_state.data_file]
shots = _load(str(data_path), data_path.stat().st_mtime)

teams = sorted(shots["team"].dropna().unique().tolist())
st.session_state.setdefault("selected_team", ALL_TEAMS)
st.session_state.setdefault("player_choice", ALL_PLAYERS)
st.session_state.setdefault("shot_type_filter", ALL)
st.session_state.setdefault("result_filter", ALL)
st.session_state.setdefault("show_table", False)

quarters_in_data = sorted(int(q) for q in shots["quarter"].dropna().unique())
has_time = bool(quarters_in_data)
st.session_state.setdefault("quarter_filter", quarters_in_data)
if any(q not in quarters_in_data for q in st.session_state.quarter_filter):
    st.session_state.quarter_filter = quarters_in_data
if st.session_state.selected_team not in [ALL_TEAMS, *teams]:
    st.session_state.selected_team = ALL_TEAMS

selected_team = None if st.session_state.selected_team == ALL_TEAMS else st.session_state.selected_team
team_shots = shots if selected_team is None else shots[shots["team"] == selected_team]
player_options = [ALL_PLAYERS, *sorted(team_shots["name"].dropna().unique().tolist())]
if st.session_state.player_choice not in player_options:
    st.session_state.player_choice = ALL_PLAYERS

# ----------------------------------------------------------------- banner
team_col, player_col, file_col = st.columns([2, 2, 5], gap="small", vertical_alignment="center")
with team_col:
    with st.popover(f"Team: {st.session_state.selected_team}", width="stretch"):
        st.radio(
            "Team",
            [ALL_TEAMS, *teams],
            key="selected_team",
            on_change=_on_team_change,
            label_visibility="collapsed",
        )
with player_col:
    with st.popover(f"Player: {st.session_state.player_choice}", width="stretch"):
        st.radio(
            "Player",
            player_options,
            key="player_choice",
            label_visibility="collapsed",
        )
with file_col:
    st.selectbox("Data", list(file_labels), key="data_file", label_visibility="collapsed")

selected_player = (
    None if st.session_state.player_choice == ALL_PLAYERS else st.session_state.player_choice
)

shot_type_options = [ALL, *sorted(shots["shot_type"].dropna().unique().tolist())]
if st.session_state.shot_type_filter not in shot_type_options:
    st.session_state.shot_type_filter = ALL

# Time window: None = no filter (all quarters / full clock), so data without a
# time column still shows everything.
chosen_quarters = st.session_state.quarter_filter
quarters = None if not has_time or not chosen_quarters or len(chosen_quarters) == len(quarters_in_data) \
    else list(chosen_quarters)
# The slider gets its full-range default via `value=` (that is what makes it a
# range slider), so the key is only in session state after the first render.
clock_high, clock_low = (CLOCK_TO_S[c] for c in st.session_state.get("clock_filter", FULL_CLOCK))
clock_range = None if not has_time or (clock_high, clock_low) == (QUARTER_S, 0) else (clock_high, clock_low)
time_filters = {"quarters": quarters, "clock_range": clock_range}

filtered = filter_shots(
    shots,
    team=selected_team,
    player=selected_player,
    shot_type=st.session_state.shot_type_filter,
    result=st.session_state.result_filter,
    **time_filters,
)
# Game stats follow team / player / time window; shot type and result only
# narrow the chart and the table.
metrics = shot_metrics(filter_shots(shots, team=selected_team, player=selected_player, **time_filters))

# ------------------------------------------------------------------- body
info_col, court_col = st.columns([1, 2], gap="small")

with info_col:
    details = []
    if selected_player:
        player_rows = team_shots[team_shots["name"] == selected_player]
        player_team = selected_team or (str(player_rows["team"].iloc[0]) if len(player_rows) else "—")
        jersey = str(player_rows["player_id"].dropna().iloc[0]) if player_rows["player_id"].notna().any() else ""
        heading = selected_player
        team_logo = logo_uri(player_team)
        team_badge = f'<img class="logo-inline" src="{team_logo}" alt="">' if team_logo else ""
        details.append(f"<div><strong>Team</strong> {team_badge}{player_team}</div>")
        if jersey:
            details.append(f"<div><strong>No.</strong> #{jersey}</div>")
        details.append(f"<div><strong>Points</strong> {metrics['points']} "
                       f"<span style='color:{colors.muted}'>(field goals)</span></div>")
        headshot = headshot_uri(player_team, jersey)
        avatar = _placeholder_svg()
    else:
        headshot = None
        heading = selected_team or ALL_TEAMS
        logos = [uri for uri in (logo_uri(t) for t in ([selected_team] if selected_team else teams)) if uri]
        avatar = ("".join(f'<img class="team-logo" src="{uri}" alt="">' for uri in logos)
                  if logos else _placeholder_svg())
    details.append(f"<div><strong>Shots</strong> {metrics['fgm']}/{metrics['fga']}</div>")
    details.append(f"<div><strong>FG%</strong> {_pct(metrics['fgm'], metrics['fga'])}</div>")
    details.append(f"<div><strong>3P%</strong> {_pct(metrics['tpm'], metrics['tpa'])} "
                   f"({metrics['tpm']}/{metrics['tpa']})</div>")
    details.append(f"<div><strong>2P</strong> {metrics['twopm']}/{metrics['twopa']}</div>")
    details.append(f"<div><strong>Layups</strong> {metrics['layupm']}/{metrics['layupa']}</div>")

    if headshot:
        # Player with a photo: big headshot, name + stats to its right.
        panel_html = f"""
        <div class="info-panel">
          <div class="player-card">
            <img class="headshot" src="{headshot}" alt="{heading}">
            <div>
              <p class="info-name">{heading}</p>
              <div class="info-meta">{"".join(details)}</div>
            </div>
          </div>
        </div>
        """
    else:
        panel_html = f"""
        <div class="info-panel">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.35rem;">
            {avatar}
            <p class="info-name">{heading}</p>
          </div>
          <div class="info-meta">{"".join(details)}</div>
        </div>
        """
    st.markdown(panel_html, unsafe_allow_html=True)

    with st.container(key="shot-filters"):
        st.selectbox("Shot type", shot_type_options, key="shot_type_filter")
        st.selectbox("Result", [ALL, "Made", "Missed"], key="result_filter")
        if has_time:
            st.pills(
                "Quarter",
                quarters_in_data,
                selection_mode="multi",
                format_func=quarter_label,
                key="quarter_filter",
            )
            st.select_slider("Game clock", CLOCK_OPTIONS, value=FULL_CLOCK, key="clock_filter")
        else:
            st.caption("No quarter / game-clock data in this file: time filter disabled.")
        st.toggle("Show shots table", key="show_table")
        st.caption("Player names are random roster picks (no stable ID yet; OCR will replace them).")

with court_col:
    fig = build_shot_chart(filtered, colors=colors, title=None, height=560)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
    )
    if st.session_state.show_table and len(filtered):
        table = filtered.sort_values(["quarter", "clock_s"], ascending=[True, False])
        st.dataframe(
            pd.DataFrame({
                "Time": [time_label(q, c) for q, c in zip(table["quarter"], table["clock_s"])],
                "Player": table["name"],
                "No.": table["player_id"],
                "Team": table["team"],
                "Type": table["shot_type"],
                "Result": table["result"],
                "Dist (ft)": table["distance"].round(1),
                "Confidence": table["confidence"] * 100,
            }),
            hide_index=True,
            width="stretch",
            column_config={
                "Confidence": st.column_config.NumberColumn(format="%.0f%%"),
            },
        )
    elif st.session_state.show_table:
        st.caption("No shots match the current filters.")
