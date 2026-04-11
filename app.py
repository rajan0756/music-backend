"""
=============================================================
  🎵 Music Recommendation System — Flask API
  Run: py app.py
  API will be available at: http://localhost:5000
=============================================================
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, origins="*")

# ─────────────────────────────────────────────
# AUDIO FEATURES & GENRE/LANGUAGE GROUPS
# ─────────────────────────────────────────────

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

GENRE_GROUPS = {
    "indian":     ["indian", "bollywood", "desi", "pop-film", "filmi", "carnatic", "hindi", "punjabi"],
    "pop":        ["pop", "indie-pop", "synth-pop", "power-pop", "electropop"],
    "asian-pop":  ["mandopop", "cantopop", "k-pop", "j-pop", "c-pop"],
    "rock":       ["rock", "alt-rock", "hard-rock", "punk-rock", "indie", "alternative", "metalcore"],
    "hiphop":     ["hip-hop", "rap", "trap", "r-n-b", "soul"],
    "electronic": ["electronic", "edm", "techno", "house", "dance", "electro", "dubstep"],
    "classical":  ["classical", "ambient", "piano", "new-age"],
    "jazz":       ["jazz", "blues", "soul", "funk"],
    "latin":      ["latin", "salsa", "reggaeton", "samba", "bossa-nova", "spanish"],
    "metal":      ["metal", "heavy-metal", "death-metal", "black-metal"],
    "country":    ["country", "bluegrass", "folk", "acoustic"],
}

# Language detection based on genre
LANGUAGE_MAP = {
    "indian":     "hindi",
    "asian-pop":  "asian",
    "latin":      "spanish",
    "pop":        "english",
    "rock":       "english",
    "hiphop":     "english",
    "electronic": "english",
    "classical":  "instrumental",
    "jazz":       "english",
    "metal":      "english",
    "country":    "english",
}

def get_genre_group(genre):
    if not isinstance(genre, str):
        return "other"
    genre = genre.lower()
    for group, genres in GENRE_GROUPS.items():
        if any(g in genre for g in genres):
            return group
    return genre

def get_language(genre_group):
    return LANGUAGE_MAP.get(genre_group, "other")
INDIAN_ARTISTS = [
    "ap dhillon", "sidhu", "shubh", "diljit", "arijit", "atif",
    "jubin", "badshah", "yo yo honey singh", "guru randhawa",
    "shreya ghoshal", "neha kakkar", "tegi pannu", "manni sandhu"
]

def fix_language(row):
    lang = get_language(row["genre_group"])
    artist = str(row["artists"]).lower()
    if any(a in artist for a in INDIAN_ARTISTS):
        return "hindi"
    return lang


# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────

print("⏳ Loading dataset...")
df = pd.read_csv("tracks_clean.csv")
df = df.dropna(subset=AUDIO_FEATURES + ["track_name", "artists"])
df = df.drop_duplicates(subset=["track_name", "artists"])
df = df.reset_index(drop=True)

scaler = MinMaxScaler()
df[AUDIO_FEATURES] = scaler.fit_transform(df[AUDIO_FEATURES])
df["genre_group"] = df["track_genre"].apply(get_genre_group)
df["language"] = df.apply(fix_language, axis=1)

print(f"✅ Dataset ready: {len(df):,} tracks loaded\n")


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({"message": "🎵 Music Recommender API is running!"})


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Please provide a search query"}), 400

    mask = df["track_name"].str.lower().str.contains(query.lower(), na=False)
    results = df[mask][["track_name", "artists", "track_genre", "popularity", "language"]].head(8)

    if results.empty:
        return jsonify({"results": [], "message": "No songs found"})

    return jsonify({"results": results.to_dict(orient="records")})


@app.route("/recommend", methods=["GET"])
def recommend():
    track_name   = request.args.get("track", "").strip()
    artist_name  = request.args.get("artist", "").strip()
    top_n        = int(request.args.get("n", 10))
    same_lang    = request.args.get("same_language", "true").lower() == "true"

    if not track_name or not artist_name:
        return jsonify({"error": "Please provide both track and artist"}), 400

    # Find seed track
    mask = (
        df["track_name"].str.lower().str.contains(track_name.lower(), na=False)
        & df["artists"].str.lower().str.contains(artist_name.lower(), na=False)
    )
    matches = df[mask]

    if matches.empty:
        return jsonify({"error": f"Track '{track_name}' by '{artist_name}' not found"}), 404

    idx  = matches.index[0]
    seed = df.loc[idx]
    seed_genre_group = seed["genre_group"]
    seed_language    = seed["language"]

    # Filter pool by genre group AND language
    if same_lang:
        pool = df[
            (df["genre_group"] == seed_genre_group) &
            (df["language"] == seed_language)
        ].copy()
    else:
        pool = df[df["genre_group"] == seed_genre_group].copy()

    # Fallback if pool too small
    if len(pool) < top_n + 5:
        pool = df[df["language"] == seed_language].copy()
    if len(pool) < top_n + 5:
        pool = df.copy()

    # Cosine similarity
    seed_vector  = df.loc[[idx], AUDIO_FEATURES].values
    pool_vectors = pool[AUDIO_FEATURES].values
    scores = cosine_similarity(seed_vector, pool_vectors)[0]

    pool = pool.copy()
    pool["_score"] = scores
    pool = pool[pool.index != idx]
    pool = pool.sort_values("_score", ascending=False).head(top_n)

    recommendations = pool[[
        "track_name", "artists", "track_genre", "popularity", "language"
    ]].copy()
    recommendations.insert(0, "similarity_score", pool["_score"].round(4).values)

    return jsonify({
        "seed": {
            "track_name": seed["track_name"],
            "artists":    seed["artists"],
            "genre":      seed.get("track_genre", "N/A"),
            "popularity": int(seed.get("popularity", 0)),
            "language":   seed_language,
        },
        "recommendations": recommendations.to_dict(orient="records")
    })


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting Music Recommender API...")
    print("   Open http://localhost:5000 in your browser to test\n")
    app.run(debug=True, port=5000)
