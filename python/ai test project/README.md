# 🎵 Music Recommendation System

A production-ready, content-based music recommendation system built with Python, scikit-learn, and Streamlit.

---

## 📌 Project Overview

This system recommends songs based on audio feature similarity. Given a single song or a listening history, it analyses 7 audio features — danceability, energy, tempo, valence, loudness, acousticness, and instrumentalness — normalises them, and uses **cosine similarity** to surface the most musically similar tracks.

The project is modular, clean, and ready to extend with collaborative filtering or a real Spotify API integration.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Single-song recommendations | Pick one song → get top 5 similar tracks instantly |
| 📜 History-based recommendations | Select multiple songs → averages feature vectors → finds your taste profile |
| 🔬 Feature-level explanations | See *why* each song was recommended (feature similarity %) |
| 📊 Radar chart comparison | Visual audio fingerprint comparison: seed vs recommendation |
| 📈 Feature bar chart | Horizontal bar chart showing per-feature similarity |
| 🎛 Dataset stats sidebar | Song count, genre count, unique artists at a glance |
| ⚠️ Graceful error handling | Clear messages when a song is not found |

---

## 🗂 Project Structure

```
music-recommendation/
│
├── data/
│   └── songs.csv               # 90-song Spotify-like dataset
│
├── src/
│   ├── preprocess.py           # Load, clean, normalise features
│   └── recommender.py          # Cosine similarity, recommend(), explain()
│
├── app.py                      # Streamlit UI
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧠 How It Works

```
songs.csv
    │
    ▼
preprocess.py
 ├── load_data()          → reads CSV, raises FileNotFoundError if missing
 ├── clean_data()         → fills NaNs with median, drops duplicate songs
 ├── extract_features()   → selects 7 audio feature columns
 └── normalize_features() → StandardScaler (zero mean, unit variance)
    │
    ▼ feature_matrix (n_songs × 7)
    │
recommender.py
 ├── recommend(song)              → cosine_similarity → top-5 songs
 ├── recommend_from_history(list) → mean(feature vectors) → cosine_similarity → top-5
 └── explain_recommendation()     → per-feature similarity % for UI explanation
    │
    ▼
app.py  (Streamlit)
 ├── Tab 1: Single Song  → selectbox → Recommend button → cards + charts
 └── Tab 2: History      → multiselect → Recommend button → cards + taste profile
```

### Cosine Similarity

Given two song vectors **A** and **B**:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

A score of **1.0** means identical audio profiles; **0.0** means completely different.

---

## 🚀 How to Run

### 1. Clone / download the project

```bash
git clone https://github.com/yourname/music-recommendation.git
cd music-recommendation
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 🧪 Running the Backend Directly

```bash
# Test preprocessing
python src/preprocess.py

# Test recommendation engine
python src/recommender.py
```

---

## 📊 Dataset

The included `data/songs.csv` is a hand-crafted Spotify-like dataset with **90 songs** across 8 genres (pop, rock, hip-hop, soul, funk, EDM, K-pop, Latin) and the following columns:

| Column | Type | Description |
|---|---|---|
| `name` | string | Song title |
| `artist` | string | Artist name |
| `genre` | string | Genre label |
| `danceability` | float 0–1 | Suitability for dancing |
| `energy` | float 0–1 | Intensity and activity |
| `tempo` | float BPM | Speed of the track |
| `valence` | float 0–1 | Musical positiveness |
| `loudness` | float dB | Overall loudness |
| `acousticness` | float 0–1 | Acoustic confidence |
| `instrumentalness` | float 0–1 | No-vocals prediction |

To use your own dataset, replace `data/songs.csv` with any CSV that includes the columns above.

---

## 🖥 Sample Output

```
── Single-song recommendations ────────────────────────
Seed: Blinding Lights

#1  Levitating          (Dua Lipa)   — 97.2% match
#2  Dynamite            (BTS)        — 96.8% match
#3  Don't Start Now     (Dua Lipa)   — 95.1% match
#4  Bad Habits          (Ed Sheeran) — 94.7% match
#5  Butter              (BTS)        — 94.3% match
```

---

## 🛠 Tech Stack

| Tool | Role |
|---|---|
| **Python 3.10+** | Core language |
| **pandas** | Data loading & manipulation |
| **NumPy** | Vector arithmetic |
| **scikit-learn** | StandardScaler, cosine_similarity |
| **Streamlit** | Interactive web UI |
| **Matplotlib** | Bar charts & radar charts |

---

## 🔭 Future Improvements

- [ ] **Collaborative filtering** — "users who liked X also liked Y"
- [ ] **Spotify API integration** — pull real audio features via Spotipy
- [ ] **User accounts & history persistence** — save listening profiles to disk/DB
- [ ] **Hybrid model** — combine content-based + collaborative signals
- [ ] **Genre-aware filtering** — option to stay within/explore outside genres
- [ ] **Playlist generation** — chain recommendations into a full playlist
- [ ] **Embedding-based similarity** — use sentence-transformers on lyrics for richer matching
- [ ] **Export to Spotify** — one-click "Add to Playlist" via OAuth

---

## 📄 License

MIT — free to use, modify, and distribute.
