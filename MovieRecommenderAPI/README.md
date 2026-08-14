# Movie Recommender API

A FastAPI backend around your TF-IDF content-based recommender, with API key auth and a "where to watch" lookup — plus a small web UI to try it out.

## Read this first — what "fetches from Netflix/JioHotstar/PrimeVideo and redirects" actually means here

There's no API — free, paid, or otherwise — that lets an outside app jump straight into playing a specific title inside Netflix, JioHotstar, or Prime Video. Those platforms require an authenticated in-app session, and none of them expose a public "play this exact movie" link. Building something that pretended to do that would either not work or rely on scraping in a way that breaks their terms of service — not something worth shipping.

What this app actually does, and what real "where to watch" features (Google's search box, JustWatch, TMDB) all do under the hood: it looks up which platforms carry a title via **TMDB's official watch-providers endpoint** (their data comes from JustWatch), and links out to that provider's own site or search page. One click closer, on the real platform — just not an auto-play deep link, because that doesn't exist anywhere.

## What's included

```
.
├── app/
│   ├── main.py           # FastAPI app: recommend + watch endpoints, API key auth
│   └── static/
│       └── index.html    # single-page UI
├── models/                # put your df.pkl, indices.pkl, tfidf.pkl, tfidf_matrix.pkl here
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Copy your pickle files** from Colab into `models/`:
   `df.pkl`, `indices.pkl`, `tfidf.pkl`, `tfidf_matrix.pkl`

2. **Get a free TMDB API key** (takes 2 minutes, no cost):
   - Sign up at https://www.themoviedb.org
   - Settings → API → Create → choose "Developer" → fill the short form
   - Copy your API Key (v3 auth)

3. **Create your `.env` file:**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   ```
   APP_API_KEY=pick-any-random-string-here
   TMDB_API_KEY=your-tmdb-key-from-step-2
   WATCH_REGION=IN
   ```

4. **Update the frontend's API key** — open `app/static/index.html`, find this line near the bottom:
   ```js
   const API_KEY = "changeme-dev-key";
   ```
   and change it to match `APP_API_KEY` from your `.env`.

5. **Install dependencies and run:**
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

6. Open **http://127.0.0.1:8000** in your browser.

## API endpoints

All endpoints (except `/`) require an `X-API-Key` header matching your `APP_API_KEY`.

### `GET /api/recommend?title={title}&n=10`
Returns the top-N most similar movies using your existing TF-IDF cosine similarity logic.

```bash
curl -H "X-API-Key: your-key" "http://127.0.0.1:8000/api/recommend?title=Toy%20Story&n=5"
```

### `GET /api/watch?title={title}`
Looks up the movie on TMDB and returns which platforms carry it in your configured region, with links out to each.

```bash
curl -H "X-API-Key: your-key" "http://127.0.0.1:8000/api/watch?title=Toy%20Story"
```

### `GET /api/health`
No auth required — quick check that models and TMDB key are loaded correctly.

## How the API key works

This protects *your* API from being called by anyone who doesn't have the key — a simple header check in `verify_api_key()` in `main.py`. It's separate from the TMDB API key, which authenticates *your server* to TMDB's API. Two different keys, two different jobs:

| Key | Protects | Set in |
|---|---|---|
| `APP_API_KEY` | Your API, from random callers | `.env`, and the frontend JS |
| `TMDB_API_KEY` | Your server's requests to TMDB | `.env` only (never exposed to the browser) |

## Notes on the dataset

Your notebook used `the-movies-dataset` (Kaggle) and titles must match **exactly** (case-sensitive) since `indices` is built directly from `df['title']`. Worth adding a fuzzy/partial-match search later — a good next step once the basics are working.

## Future improvements

- Fuzzy title matching (e.g. `difflib` or a proper search-as-you-type) instead of exact match
- Cache TMDB responses (they don't change often) to cut down repeated calls
- Add poster images to the recommendation grid, not just the selected movie
- Deploy the backend somewhere reachable (Render, Railway, Fly.io) so the frontend isn't stuck on localhost
