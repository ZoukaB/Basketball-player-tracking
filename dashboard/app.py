"""Basketball dashboard home."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from lib.colors import load_colors

st.set_page_config(
    page_title="Basketball Dashboard",
    layout="wide",
)

colors = load_colors()
st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {colors.background}; }}
        h1, h2, h3, p, label {{ color: {colors.text}; font-family: {colors.font}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Basketball Dashboard")
st.write(
    "Interactive views of tracked games. Start with the shot chart generator, "
    "built from mock shot locations on an NBA half-court (47 × 50 ft)."
)
st.page_link("pages/1_Shot_Chart.py", label="Open Shot Chart")
