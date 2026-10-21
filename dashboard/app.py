"""Live shot chart dashboard.

A background `LiveShotSession` (src/live.py) runs the shot pipeline over the
video source; fragments poll it so the frame, metrics and court chart update
while the Team / Player / Type / Result filters stay interactive.

Run (from repo root):
    .venv/Scripts/python.exe -m streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parent
sys.path.insert(0, str(DASHBOARD_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

import streamlit as st

from lib.colors import load_colors, theme_css
from lib.data import filter_shots, rows_to_frame, shot_metrics
from lib.shot_chart import build_shot_chart

ALL_TEAMS = "All teams"
ALL_PLAYERS = "All players"
ALL = "All"
SHOT_TYPES = [ALL, "2pt", "3pt", "layup"]
RESULTS = [ALL, "Made", "Missed"]

st.set_page_config(
    page_title="Live Shot Chart",
    layout="wide",
    initial_sidebar_state="collapsed",
)

colors = load_colors()
st.markdown(theme_css(colors), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_session():
    """One live session (and one set of models) per Streamlit server.

    Cheap: src.live defers the heavy imports; models load in a background thread.
    """
    from src.live import LiveShotSession
    from src.utils import load_config

    cfg = load_config()
    session = LiveShotSession(cfg)
    session.preload()
    return cfg, session


cfg, session = get_session()
rosters = cfg["teams"].get("rosters") or {}
teams = sorted(rosters) or sorted(cfg["teams"]["team_names"].values())


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


def _player_options(team: str | None) -> list[str]:
    """Roster names (random identities are drawn from it), plus any name seen in shots."""
    names = set()
    for team_name, roster in rosters.items():
        if team is None or team_name == team:
            names.update(roster.values())
    rows = session.snapshot()["rows"]
    names.update(r["name"] for r in rows if r.get("name") and (team is None or r["team"] == team))
    return [ALL_PLAYERS, *sorted(names)]


# ------------------------------------------------------------------ state
st.session_state.setdefault("selected_team", ALL_TEAMS)
st.session_state.setdefault("player_choice", ALL_PLAYERS)
st.session_state.setdefault("shot_type_filter", ALL)
st.session_state.setdefault("result_filter", ALL)
st.session_state.setdefault(
    "source", cfg["input"]["video"] if (REPO_ROOT / cfg["input"]["video"]).exists() else "data"
)
if st.session_state.selected_team not in [ALL_TEAMS, *teams]:
    st.session_state.selected_team = ALL_TEAMS

selected_team = None if st.session_state.selected_team == ALL_TEAMS else st.session_state.selected_team
player_options = _player_options(selected_team)
if st.session_state.player_choice not in player_options:
    st.session_state.player_choice = ALL_PLAYERS

# ----------------------------------------------------------------- banner
team_col, player_col, source_col, start_col, pause_col, stop_col = st.columns(
    [2, 2, 4, 1, 1, 1], gap="small", vertical_alignment="center"
)
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
with source_col:
    st.text_input(
        "Source",
        key="source",
        placeholder="video file, folder of clips, camera index or URL",
        label_visibility="collapsed",
        disabled=session.running,
    )
with start_col:
    if st.button("Start", width="stretch", disabled=session.running):
        session.start(st.session_state.source)
        st.rerun()
with pause_col:
    paused = session.snapshot()["status"]["state"] == "paused"
    if st.button("Resume" if paused else "Pause", width="stretch", disabled=not session.running):
        session.resume() if paused else session.pause()
        st.rerun()
with stop_col:
    if st.button("Stop", width="stretch", disabled=not session.running):
        session.stop()
        session.join(timeout=5.0)
        st.rerun()

selected_player = (
    None if st.session_state.player_choice == ALL_PLAYERS else st.session_state.player_choice
)

# ------------------------------------------------------------------- body
video_col, chart_col = st.columns([3, 2], gap="small")


@st.fragment(run_every=0.25)
def live_frame() -> None:
    snap = session.snapshot()
    status = snap["status"]
    if snap["jpeg"]:
        st.image(snap["jpeg"], width="stretch")
    elif status["state"] == "idle":
        st.info("Press Start to process the source.")
    elif status["state"] in ("loading models", "warming up team classifier"):
        st.info(f"{status['state'].capitalize()} in the background (a few minutes on first launch). "
                "You can press Start now; it begins as soon as they are ready.")
    else:
        st.info(f"{status['state'].capitalize()}...")
    clip = f"clip {status['clip_index']}/{status['n_clips']} · {status['clip']}" if status["clip"] else ""
    teams_s = "fitted" if status["teams_fitted"] else "bootstrapping"
    line = (f"<b>{status['state']}</b> · {clip} · frame {status['frame']} · "
            f"{status['fps']:.1f} fps · teams {teams_s} · {len(snap['rows'])} shots")
    if status["error"]:
        line += f" · <span style='color:#B00020'>{status['error']}</span>"
    st.markdown(f'<div class="live-status">{line}</div>', unsafe_allow_html=True)


@st.fragment(run_every=1.0)
def live_chart() -> None:
    shots = rows_to_frame(session.snapshot()["rows"])
    filtered = filter_shots(
        shots,
        team=selected_team,
        player=selected_player,
        shot_type=st.session_state.shot_type_filter,
        result=st.session_state.result_filter,
    )
    metrics = shot_metrics(filtered)
    fg = f"{metrics['fg_pct'] * 100:.1f}%" if metrics["fga"] else "—"
    tp = f"{metrics['tp_pct'] * 100:.1f}%" if metrics["tpa"] else "—"

    if selected_player:
        player_rows = shots[shots["name"] == selected_player]
        if selected_team:
            player_rows = player_rows[player_rows["team"] == selected_team]
        jersey = str(player_rows["player_id"].dropna().iloc[0]) if player_rows["player_id"].notna().any() else ""
        team_line = selected_team or (str(player_rows["team"].iloc[0]) if len(player_rows) else "—")
        heading = selected_player
    else:
        heading = selected_team or ALL_TEAMS
        team_line, jersey = None, ""

    details = []
    if selected_player:
        details.append(f"<div><strong>Team</strong> {team_line}</div>")
        if jersey:
            details.append(f"<div><strong>No.</strong> #{jersey}</div>")
    details.append(f"<div><strong>Shots</strong> {metrics['fga']} ({metrics['fgm']} made)</div>")
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

    fig = build_shot_chart(filtered, colors=colors, title=None, height=560)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
    )

    if len(filtered):
        recent = filtered.sort_values(["video", "frame"]).tail(8).iloc[::-1]
        st.dataframe(
            recent[["frame", "team", "player_id", "name", "shot_type", "result", "distance"]],
            hide_index=True,
            width="stretch",
        )


with video_col:
    st.markdown('<div class="live-frame"></div>', unsafe_allow_html=True)
    live_frame()
    type_col, result_col = st.columns(2, gap="small")
    with type_col:
        st.selectbox("Shot type", SHOT_TYPES, key="shot_type_filter")
    with result_col:
        st.selectbox("Result", RESULTS, key="result_filter")
    st.caption("Player names are random roster picks (no stable ID yet; OCR will replace them).")

with chart_col:
    live_chart()
