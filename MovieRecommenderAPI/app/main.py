"""
Movie Recommender API
----------------------
Serves content-based recommendations from your existing TF-IDF pickle files,
protected by a simple API key, plus a "where to watch" lookup powered by
TMDB's official watch-providers endpoint (sourced from JustWatch data).

IMPORTANT — read this before wiring up the frontend:
There is no public API (free or paid) that lets you jump straight into
playing a specific title inside Netflix / JioHotstar / Prime Video from an
outside app — those platforms require an authenticated in-app session and
don't expose that kind of deep link. What TMDB *does* give us is an accurate
"which platforms carry this title, in this region" answer, plus a link to
each platform's own site. That's what /watch below returns. See README.md.
"""

import os
import pickle
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import requests
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
API_KEY = os.environ.get("APP_API_KEY", "changeme-dev-key")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")  # get a free one at themoviedb.org
TMDB_BASE = "https://api.themoviedb.org/3"
WATCH_REGION = os.environ.get("WATCH_REGION", "IN")  # ISO 3166-1 country code

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Best-effort search links for each provider (NOT deep links to the exact
# title's play page — see the docstring above for why that's not possible).
PROVIDER_SEARCH_URLS = {
    "Netflix": "https://www.netflix.com/search?q={query}",
    "Amazon Prime Video": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={query}",
    "Disney Plus Hotstar": "https://www.hotstar.com/in/search?q={query}",
    "JioHotstar": "https://www.hotstar.com/in/search?q={query}",
    "Jio Cinema": "https://www.jiocinema.com/search/{query}",
}

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(title="Movie Recommender API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Load your existing pickles (from the Colab notebook)
# ---------------------------------------------------------------------
def load_pickle(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing {filename} in models/. Copy your df.pkl, indices.pkl, "
            f"tfidf.pkl, and tfidf_matrix.pkl from Colab into the models/ folder."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


df = None
indices = None
tfidf = None
tfidf_matrix = None


@app.on_event("startup")
def load_models():
    global df, indices, tfidf, tfidf_matrix
    df = load_pickle("df.pkl") if os.path.exists(os.path.join(MODELS_DIR, "df.pkl")) else None
    indices = load_pickle("indices.pkl") if os.path.exists(os.path.join(MODELS_DIR, "indices.pkl")) else None
    tfidf = load_pickle("tfidf.pkl") if os.path.exists(os.path.join(MODELS_DIR, "tfidf.pkl")) else None
    tfidf_matrix = (
        load_pickle("tfidf_matrix.pkl") if os.path.exists(os.path.join(MODELS_DIR, "tfidf_matrix.pkl")) else None
    )


# ---------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key (X-API-Key header)")
    return True


# ---------------------------------------------------------------------
# Recommendation logic (straight from your notebook)
# ---------------------------------------------------------------------
def get_recommendations(title: str, n: int = 10):
    from sklearn.metrics.pairwise import cosine_similarity

    if title not in indices:
        return None

    idx = indices[title]
    sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_score.argsort()[::-1][1 : n + 1]
    return df["title"].iloc[similar_idx].tolist()


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": df is not None,
        "tmdb_configured": bool(TMDB_API_KEY),
    }


@app.get("/api/recommend")
def recommend(
    title: str = Query(..., description="Exact movie title, e.g. 'Toy Story'"),
    n: int = Query(10, ge=1, le=25),
    _: bool = None,
    x_api_key: Optional[str] = Header(None),
):
    verify_api_key(x_api_key)

    if df is None or indices is None or tfidf_matrix is None:
        raise HTTPException(status_code=503, detail="Model files not loaded — check models/ folder")

    results = get_recommendations(title, n)
    if results is None:
        raise HTTPException(status_code=404, detail=f"'{title}' not found in dataset")

    return {"query": title, "recommendations": results}


@app.get("/api/watch")
def where_to_watch(
    title: str = Query(..., description="Movie title to look up on TMDB"),
    x_api_key: Optional[str] = Header(None),
):
    verify_api_key(x_api_key)

    if not TMDB_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="TMDB_API_KEY not set. Get a free key at https://www.themoviedb.org/settings/api",
        )

    # 1. Search TMDB for the title to get its ID
    search_resp = requests.get(
        f"{TMDB_BASE}/search/movie",
        params={"api_key": TMDB_API_KEY, "query": title},
        timeout=10,
    )
    search_resp.raise_for_status()
    results = search_resp.json().get("results", [])

    if not results:
        raise HTTPException(status_code=404, detail=f"'{title}' not found on TMDB")

    movie = results[0]
    movie_id = movie["id"]

    # 2. Fetch watch providers for that movie
    providers_resp = requests.get(
        f"{TMDB_BASE}/movie/{movie_id}/watch/providers",
        params={"api_key": TMDB_API_KEY},
        timeout=10,
    )
    providers_resp.raise_for_status()
    region_data = providers_resp.json().get("results", {}).get(WATCH_REGION, {})

    flatrate = region_data.get("flatrate", [])  # subscription streaming
    rent = region_data.get("rent", [])
    buy = region_data.get("buy", [])

    def format_providers(provider_list):
        out = []
        for p in provider_list:
            name = p["provider_name"]
            search_template = PROVIDER_SEARCH_URLS.get(name)
            out.append(
                {
                    "name": name,
                    "logo": f"https://image.tmdb.org/t/p/original{p['logo_path']}",
                    "search_url": search_template.format(query=requests.utils.quote(title))
                    if search_template
                    else None,
                }
            )
        return out

    return {
        "title": movie.get("title"),
        "year": (movie.get("release_date") or "")[:4],
        "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
        "region": WATCH_REGION,
        "streaming": format_providers(flatrate),
        "rent": format_providers(rent),
        "buy": format_providers(buy),
        "tmdb_watch_page": region_data.get("link"),  # official TMDB/JustWatch page with all provider buttons
    }


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
