"""
app.py
------
Streamlit front-end for the Music Recommendation System.

Run with:
    streamlit run app.py
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── path setup so src/ imports work regardless of CWD ──────────────────────
ROOT = pathlib.Path(__file__).parent


from src.preprocess import preprocess
from src.recommender import (
    recommend,
    recommend_from_history,
    explain_recommendation,
)
# ───────────────────────────────────────────────────────────────────────────
# Config & constants
# ───────────────────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "songs.csv"
TOP_K     = 5

GENRE_EMOJI = {
    "pop":    "🎵",
    "rock":   "🎸",
    "hiphop": "🎤",
    "soul":   "🎶",
    "funk":   "🕺",
    "edm":    "🎧",
    "kpop":   "✨",
    "latin":  "💃",
}

# ───────────────────────────────────────────────────────────────────────────
# Page configuration
# ───────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────────────────
# Custom CSS — dark, refined music-player aesthetic
# ───────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0e0e14;
    --surface:   #16161f;
    --card:      #1c1c28;
    --accent:    #7c5cfc;
    --accent2:   #fc5c7d;
    --text:      #e8e8f0;
    --muted:     #6b6b85;
    --border:    #2a2a3d;
    --green:     #1db954;
}

/* ── Global overrides ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1a1028 0%, #0e0e14 60%, #12111e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(124,92,252,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(252,92,125,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(90deg, #fff 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}

/* ── Song card ── */
.song-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s;
}
.song-card:hover { border-color: var(--accent); }
.rank-badge {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: var(--border);
    min-width: 2rem;
    text-align: center;
}
.song-info { flex: 1; }
.song-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: var(--text);
    margin: 0;
}
.song-meta {
    font-size: 0.82rem;
    color: var(--muted);
    margin: 0.15rem 0 0 0;
}
.sim-pill {
    background: rgba(124,92,252,0.15);
    border: 1px solid rgba(124,92,252,0.3);
    color: var(--accent);
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    white-space: nowrap;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Selectbox & multiselect ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Syne', sans-serif !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: var(--accent) !important; }

/* ── Info/Success boxes ── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# Data loading  (cached so it only runs once per session)
# ───────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all(filepath: str):
    return preprocess(filepath)


# ───────────────────────────────────────────────────────────────────────────
# Helper rendering functions
# ───────────────────────────────────────────────────────────────────────────

def genre_badge(genre: str) -> str:
    g = str(genre).lower()
    emoji = GENRE_EMOJI.get(g, "🎼")
    return f"{emoji} {genre.title()}"


def render_song_card(rank: int, row: pd.Series):
    sim_pct  = round(row.get("similarity_score", 0) * 100, 1)
    genre    = genre_badge(row.get("genre", ""))
    artist   = row.get("artist", "Unknown")
    title    = row.get("name", "Unknown")

    st.markdown(f"""
    <div class="song-card">
        <div class="rank-badge">#{rank}</div>
        <div class="song-info">
            <p class="song-title">{title}</p>
            <p class="song-meta">{artist} &nbsp;·&nbsp; {genre}</p>
        </div>
        <div class="sim-pill">{sim_pct}% match</div>
    </div>
    """, unsafe_allow_html=True)


def render_explanation_chart(explanations: list[dict], seed: str, rec: str):
    """Render a horizontal bar chart comparing feature similarity."""
    matplotlib.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.facecolor":   "#1c1c28",
        "figure.facecolor": "#1c1c28",
        "text.color":       "#e8e8f0",
        "axes.labelcolor":  "#e8e8f0",
        "xtick.color":      "#6b6b85",
        "ytick.color":      "#e8e8f0",
        "axes.edgecolor":   "#2a2a3d",
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })

    features  = [e["feature"].replace("_", " ").title() for e in explanations]
    sim_vals  = [e["similarity_pct"] for e in explanations]
    colors    = ["#7c5cfc" if v >= 70 else "#fc5c7d" if v >= 40 else "#3a3a50" for v in sim_vals]

    fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.55)))
    bars = ax.barh(features, sim_vals, color=colors, height=0.55, edgecolor="none")

    # Value labels
    for bar, val in zip(bars, sim_vals):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%", va="center", ha="left",
            fontsize=9, color="#e8e8f0"
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Feature Similarity (%)", fontsize=9, color="#6b6b85")
    ax.set_title(f"Why '{rec}' matches '{seed}'", fontsize=10,
                 color="#e8e8f0", pad=12, fontweight="bold")
    ax.axvline(70, color="#2a2a3d", linewidth=1, linestyle="--")

    legend_patches = [
        mpatches.Patch(color="#7c5cfc", label="High (≥70%)"),
        mpatches.Patch(color="#fc5c7d", label="Medium (40-69%)"),
        mpatches.Patch(color="#3a3a50", label="Low (<40%)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=8, framealpha=0.2, labelcolor="#e8e8f0",
              facecolor="#16161f", edgecolor="#2a2a3d")

    plt.tight_layout()
    return fig


def render_radar_chart(seed_row: pd.Series, rec_row: pd.Series, features: list[str]):
    """Polar radar chart comparing seed vs recommended song audio features."""
    available = [f for f in features if f in seed_row.index and f in rec_row.index]
    if len(available) < 3:
        return None

    labels = [f.replace("_", " ").title() for f in available]
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close polygon

    def _normalize_vals(row):
        """Normalise to [0,1] using pre-known practical ranges."""
        RANGES = {
            "danceability":     (0.0, 1.0),
            "energy":           (0.0, 1.0),
            "valence":          (0.0, 1.0),
            "acousticness":     (0.0, 1.0),
            "instrumentalness": (0.0, 1.0),
            "tempo":            (40, 220),
            "loudness":         (-30, 0),
        }
        vals = []
        for f in available:
            lo, hi = RANGES.get(f, (0, 1))
            v = (float(row[f]) - lo) / (hi - lo + 1e-9)
            vals.append(min(max(v, 0.0), 1.0))
        return vals

    seed_vals = _normalize_vals(seed_row) + [_normalize_vals(seed_row)[0]]
    rec_vals  = _normalize_vals(rec_row)  + [_normalize_vals(rec_row)[0]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5),
                           subplot_kw=dict(polar=True),
                           facecolor="#1c1c28")
    ax.set_facecolor("#1c1c28")
    ax.spines["polar"].set_color("#2a2a3d")
    ax.grid(color="#2a2a3d", linewidth=0.8)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color="#e8e8f0")

    ax.plot(angles, seed_vals, "o-", linewidth=2, color="#7c5cfc")
    ax.fill(angles, seed_vals, alpha=0.18, color="#7c5cfc")

    ax.plot(angles, rec_vals, "o-", linewidth=2, color="#fc5c7d")
    ax.fill(angles, rec_vals, alpha=0.18, color="#fc5c7d")

    ax.set_title("Audio Profile Comparison", size=10,
                 color="#e8e8f0", pad=16, fontweight="bold")

    legend = ax.legend(
        [seed_row["name"][:20], rec_row["name"][:20]],
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        fontsize=8, framealpha=0.2, labelcolor="#e8e8f0",
        facecolor="#16161f", edgecolor="#2a2a3d"
    )
    plt.tight_layout()
    return fig


# ───────────────────────────────────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────────────────────────────────

def build_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown('<div class="section-label">📊 Dataset Info</div>', unsafe_allow_html=True)
        st.metric("Total Songs",    len(df))
        st.metric("Genres",         df["genre"].nunique() if "genre" in df.columns else "—")
        st.metric("Unique Artists", df["artist"].nunique() if "artist" in df.columns else "—")

        st.divider()
        st.markdown('<div class="section-label">🎛 Feature Legend</div>', unsafe_allow_html=True)
        feature_info = {
            "💃 Danceability":     "Suitability for dancing (0–1)",
            "⚡ Energy":            "Intensity & activity (0–1)",
            "🎵 Valence":           "Positiveness / happiness (0–1)",
            "🥁 Tempo":             "Speed in BPM",
            "🔊 Loudness":          "Overall loudness (dB)",
            "🎸 Acousticness":      "Acoustic confidence (0–1)",
            "🎹 Instrumentalness":  "No-vocal prediction (0–1)",
        }
        for label, desc in feature_info.items():
            st.markdown(f"**{label}**  \n<small style='color:#6b6b85'>{desc}</small>",
                        unsafe_allow_html=True)

        st.divider()
        st.markdown(
            "<small style='color:#6b6b85'>Built with ❤️ using Python, "
            "scikit-learn & Streamlit</small>",
            unsafe_allow_html=True,
        )


# ───────────────────────────────────────────────────────────────────────────
# Main app
# ───────────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ─────────────────────────────────────────────────────────
    with st.spinner("Loading music library…"):
        try:
            df, feature_matrix, feature_cols = load_all(str(DATA_PATH))
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    song_names = sorted(df["name"].tolist())

    # ── Sidebar ───────────────────────────────────────────────────────────
    build_sidebar(df)

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🎵 Music Recommendation System</h1>
        <p>Discover songs you'll love — powered by content-based filtering & cosine similarity</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode tabs ─────────────────────────────────────────────────────────
    tab_single, tab_history = st.tabs(["🎯  Single Song", "📜  Listening History"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — single song
    # ══════════════════════════════════════════════════════════════════════
    with tab_single:
        st.markdown('<div class="section-label">Select a song to find similar tracks</div>',
                    unsafe_allow_html=True)
        seed_song = st.selectbox(
            "Choose a song",
            song_names,
            index=0,
            key="single_seed",
            label_visibility="collapsed",
        )

        # Show seed song details
        seed_row = df[df["name"] == seed_song].iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💃 Danceability", f"{seed_row.get('danceability', '—'):.2f}")
        col2.metric("⚡ Energy",        f"{seed_row.get('energy',       '—'):.2f}")
        col3.metric("🎵 Valence",       f"{seed_row.get('valence',      '—'):.2f}")
        col4.metric("🥁 Tempo",         f"{seed_row.get('tempo',        '—'):.0f} BPM")

        st.write("")
        recommend_btn = st.button("🔍  Find Similar Songs", key="btn_single")

        if recommend_btn:
            with st.spinner("Finding your next favourite tracks…"):
                try:
                    results = recommend(seed_song, df, feature_matrix, top_k=TOP_K)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

            st.write("")
            st.markdown(f'<div class="section-label">Top {TOP_K} recommendations for "{seed_song}"</div>',
                        unsafe_allow_html=True)

            for i, (_, row) in enumerate(results.iterrows(), start=1):
                render_song_card(i, row)

            # ── Explanation section ────────────────────────────────────
            st.write("")
            with st.expander("🔬  Why were these songs recommended? (Feature Analysis)"):
                st.markdown(
                    "<small style='color:#6b6b85'>The system computes <b>cosine similarity</b> "
                    "on 7 normalised audio features. Below is a per-feature breakdown for the "
                    "top recommendation.</small>",
                    unsafe_allow_html=True,
                )
                top_rec = results.iloc[0]

                exp_col, radar_col = st.columns([1.2, 1])

                with exp_col:
                    explanations = explain_recommendation(
                        seed_song, top_rec["name"], df, feature_cols
                    )
                    fig_exp = render_explanation_chart(
                        explanations, seed_song[:22], top_rec["name"][:22]
                    )
                    st.pyplot(fig_exp, use_container_width=True)
                    plt.close(fig_exp)

                with radar_col:
                    rec_row  = df[df["name"] == top_rec["name"]].iloc[0]
                    fig_radar = render_radar_chart(seed_row, rec_row, feature_cols)
                    if fig_radar:
                        st.pyplot(fig_radar, use_container_width=True)
                        plt.close(fig_radar)

                # Feature-by-feature table
                st.markdown("##### Feature Similarity Table")
                exp_df = pd.DataFrame(explanations)[
                    ["feature", "seed_value", "rec_value", "similarity_pct", "description"]
                ]
                exp_df.columns = ["Feature", f"{seed_song[:18]}…", top_rec["name"][:20], "Similarity %", "What it means"]
                st.dataframe(exp_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — listening history
    # ══════════════════════════════════════════════════════════════════════
    with tab_history:
        st.markdown(
            '<div class="section-label">Select songs from your listening history</div>',
            unsafe_allow_html=True,
        )
        selected_history = st.multiselect(
            "Your listening history",
            song_names,
            default=song_names[:3],
            key="history_select",
            label_visibility="collapsed",
            placeholder="Choose 2 or more songs…",
        )

        if len(selected_history) < 2:
            st.info("ℹ️  Please select at least **2 songs** for history-based recommendations.")
        else:
            st.markdown(
                f"<small style='color:#6b6b85'>System will average the audio fingerprint of "
                f"<b>{len(selected_history)}</b> songs and find the closest matches.</small>",
                unsafe_allow_html=True,
            )
            st.write("")
            hist_btn = st.button("🎶  Get Recommendations from History", key="btn_history")

            if hist_btn:
                with st.spinner("Analysing your taste profile…"):
                    try:
                        results_h = recommend_from_history(
                            selected_history, df, feature_matrix, top_k=TOP_K
                        )
                    except ValueError as e:
                        st.error(str(e))
                        st.stop()

                st.write("")
                st.markdown(f'<div class="section-label">Top {TOP_K} picks based on your history</div>',
                            unsafe_allow_html=True)

                for i, (_, row) in enumerate(results_h.iterrows(), start=1):
                    render_song_card(i, row)

                # ── Taste Profile visualisation ────────────────────────
                st.write("")
                with st.expander("🧬  Your Taste Profile"):
                    st.markdown(
                        "<small style='color:#6b6b85'>Your <b>taste centroid</b> — the "
                        "average audio fingerprint across your selected songs.</small>",
                        unsafe_allow_html=True,
                    )
                    available_feats = [f for f in feature_cols if f in df.columns]
                    history_df  = df[df["name"].isin(selected_history)]
                    centroid     = history_df[available_feats].mean()

                    # Radar for centroid vs top recommendation
                    if not results_h.empty:
                        top_h  = results_h.iloc[0]
                        rec_h_row = df[df["name"] == top_h["name"]].iloc[0]
                        centroid_series = centroid.rename("Your Taste Avg")
                        centroid_series["name"] = "Your Taste Profile"

                        fig_r2 = render_radar_chart(centroid_series, rec_h_row, available_feats)
                        if fig_r2:
                            st.pyplot(fig_r2, use_container_width=True)
                            plt.close(fig_r2)

                    st.markdown("##### Your Average Audio Features")
                    centroid_display = centroid.round(3).to_frame(name="Average Value")
                    centroid_display.index = centroid_display.index.str.replace("_", " ").str.title()
                    st.dataframe(centroid_display, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
