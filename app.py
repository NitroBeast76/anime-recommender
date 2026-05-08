# ------------------------------------------------------------
# Import necessary libraries
# ------------------------------------------------------------
from collections import defaultdict
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from urllib.parse import unquote

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import os
import pickle
import json
import requests
from dotenv import load_dotenv
import logging

# ------------------------------------------------------------
# Set up logging (replaces print statements)
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Autocomplete cache (time‑based)
# ------------------------------------------------------------
autocomplete_cache = {}  # { query: (titles, timestamp) }
CACHE_TTL_SECONDS = 300  # 5 minutes

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------
# Initialize the Flask application
# ------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------
# How many recommendations to show per page
# ------------------------------------------------------------
RESULTS_PER_PAGE = 20

# ------------------------------------------------------------
# Ollama configuration (for LLM-based matching)
# ------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")  # small, fast, local
MAX_LLM_CANDIDATES = 60  # reduced from 300 to avoid context overflow

# ------------------------------------------------------------
# Misspell cache (loaded from JSON file)
# ------------------------------------------------------------
CACHE_FILE = os.getenv("MISSPELL_CACHE", "misspell_cache.json")


def load_misspell_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"Loaded {len(cache)} cached misspellings.")
        return cache
    logger.info("No misspell cache file found – starting empty.")
    return {}


def save_misspell_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(cache)} misspellings to cache.")


misspell_cache = load_misspell_cache()

# ------------------------------------------------------------
# Load the embedding model
# ------------------------------------------------------------
logger.info("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model ready.")


# ------------------------------------------------------------
# Load and verify anime data and feature matrix
# ------------------------------------------------------------
def load_and_verify(
    anime_file="anime_data.csv", feature_file="anime_features_scaled.csv"
):
    """
    Loads both the anime dataset and the feature matrix, verifies they are aligned,
    and builds efficient lookup dictionaries.
    """
    try:
        logger.info(f"Loading anime dataset from {anime_file}...")
        anime_df = pd.read_csv(anime_file, encoding="utf-8")
        anime_df["cluster_id"] = anime_df["cluster_id"].astype(int)
    except Exception as e:
        logger.error(f"Failed to load {anime_file}: {e}")
        raise

    try:
        logger.info(f"Loading feature matrix from {feature_file}...")
        feature_df = pd.read_csv(feature_file, encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to load {feature_file}: {e}")
        raise

    # Sort both alphabetically by title
    anime_df = anime_df.sort_values(by="title").reset_index(drop=True)

    if "title" not in feature_df.columns:
        raise ValueError(
            f"Feature matrix '{feature_file}' has no 'title' column. "
            "Alignment cannot be verified. Please regenerate the features with titles."
        )

    feature_df = feature_df.sort_values(by="title").reset_index(drop=True)
    mismatches = (anime_df["title"].values != feature_df["title"].values).sum()
    if mismatches > 0:
        raise ValueError(
            f"ALIGNMENT ERROR: {mismatches} rows do not match between "
            f"{anime_file} and {feature_file}."
        )
    logger.info("Alignment check passed — both files are in the same order.")

    # Drop title column from features (keep only numeric features)
    feature_df = feature_df.drop(columns=["title"])

    # Check for NaN in feature matrix
    if feature_df.isna().any().any():
        logger.warning("Feature matrix contains NaN values. Filling with 0.")
        feature_df = feature_df.fillna(0)

    anime_data = anime_df.to_dict(orient="records")
    feature_matrix = feature_df.values

    # Build lookup maps for O(1) access
    title_to_index = {item["title"]: idx for idx, item in enumerate(anime_data)}
    cluster_to_indices = {}
    for idx, item in enumerate(anime_data):
        cid = item["cluster_id"]
        cluster_to_indices.setdefault(cid, []).append(idx)

    logger.info(f"Loaded {len(anime_data)} anime titles.")
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    logger.info(f"Clusters: {len(cluster_to_indices)}")

    return anime_data, feature_matrix, title_to_index, cluster_to_indices


anime_data, feature_matrix, title_to_index, cluster_to_indices = load_and_verify()


# ------------------------------------------------------------
# Build or load combined title embeddings
# ------------------------------------------------------------
def load_title_embeddings(anime_data, cache_file="title_embeddings.pkl"):
    """
    Embeds a combined text of title + english_title + japanese_title for each anime.
    """
    if os.path.exists(cache_file):
        logger.info("Loading cached title embeddings...")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        # Verify cache structure
        if isinstance(cache, dict) and "titles" in cache and "embeddings" in cache:
            current_titles = [item["title"] for item in anime_data]
            if cache["titles"] == current_titles:
                logger.info(
                    f"Embeddings loaded from cache ({len(cache['titles'])} titles)."
                )
                return cache["titles"], cache["embeddings"]
            else:
                logger.info("Dataset has changed — rebuilding embeddings cache...")
        else:
            logger.info("Cache file has invalid format — rebuilding...")

    logger.info("Building combined title embeddings...")
    combined_texts = []
    for item in anime_data:
        parts = [item["title"]]
        if "english_title" in item and str(item["english_title"]).lower() != "unknown":
            parts.append(str(item["english_title"]))
        if (
            "japanese_title" in item
            and str(item["japanese_title"]).lower() != "unknown"
        ):
            parts.append(str(item["japanese_title"]))
        combined = " | ".join(parts)
        combined_texts.append(combined)

    titles = [item["title"] for item in anime_data]
    embeddings = embedding_model.encode(
        combined_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )

    with open(cache_file, "wb") as f:
        pickle.dump({"titles": titles, "embeddings": embeddings}, f)

    logger.info(f"Embeddings built and cached ({len(titles)} titles).")
    return titles, embeddings


title_list, title_embeddings = load_title_embeddings(anime_data)

# ------------------------------------------------------------
# Build prefix index for lightning‑fast autocomplete
# ------------------------------------------------------------
logger.info("Building prefix index for autocomplete...")
prefix_index = defaultdict(list)
for title in title_list:
    lower = title.lower()
    for i in range(2, len(lower) + 1):
        prefix = lower[:i]
        prefix_index[prefix].append(title)

# Deduplicate and sort each entry alphabetically
for prefix in prefix_index:
    prefix_index[prefix] = sorted(list(set(prefix_index[prefix])))
logger.info(f"Prefix index built with {len(prefix_index)} entries.")


# ------------------------------------------------------------
# Helper: get top N closest titles via embeddings
# ------------------------------------------------------------
def get_top_n_titles(query, n=300):
    """Return the top n titles (as list) sorted by cosine similarity."""
    n = min(n, len(title_list))  # cap to available titles
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, title_embeddings)[0]
    top_indices = np.argpartition(scores, -n)[-n:]
    top_scores = scores[top_indices]
    sorted_idx = top_indices[np.argsort(top_scores)[::-1]]
    return [title_list[i] for i in sorted_idx]


# ------------------------------------------------------------
# LLM matching with funneled list (capped candidates)
# ------------------------------------------------------------
def match_title_with_llm(user_input, max_candidates=MAX_LLM_CANDIDATES):
    """
    1. Check misspell cache.
    2. If not found, retrieve top-{max_candidates} titles via embeddings.
    3. Send those candidates to the LLM (Ollama) to pick the correct one.
    4. If LLM fails or returns an invalid title, fallback to the top embedding match.
    Returns the matched title (str) or None.
    """
    query = user_input.strip()
    if not query:
        return None

    # ---- Cache check ----
    cached = misspell_cache.get(query.lower())
    if cached and cached in title_to_index:
        logger.info(f"Cache hit: '{query}' → '{cached}'")
        return cached
    elif cached:
        logger.warning(
            f"Cache hit for '{query}' → '{cached}' but title not in dataset – ignoring cache."
        )

    # ---- Get top candidates ----
    candidates = get_top_n_titles(query, max_candidates)
    if not candidates:
        return None

    # ---- Build LLM prompt ----
    titles_formatted = "\n".join(f"- {t}" for t in candidates)
    prompt = f"""You are an anime title matcher. Given a user query (which may be misspelled, abbreviated, or in a foreign language), find the closest matching official English title from the list below. Return ONLY the exact title as it appears in the list. If no reasonable match exists, return exactly "NOT_FOUND".

List of valid titles:
{titles_formatted}

User query: {query}

Your answer (only the title or NOT_FOUND):"""

    logger.info(f"Query: '{query}' – sending {len(candidates)} candidates to LLM...")

    # ---- Call Ollama ----
    answer = None
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=15,
        )
        data = response.json()
        answer = data["response"].strip().strip('"').strip("'")
    except Exception as e:
        logger.error(f"LLM error: {e}")

    # ---- Process LLM answer ----
    if answer == "NOT_FOUND":
        logger.info("LLM returned NOT_FOUND.")
        return None

    # Verify the returned title
    if answer in candidates:
        logger.info(f"LLM matched: '{answer}'")
        # Update misspell cache for future queries
        if query.lower() not in misspell_cache:
            misspell_cache[query.lower()] = answer
            save_misspell_cache(misspell_cache)
        return answer

    # Case‑insensitive fallback
    for candidate in candidates:
        if answer and answer.lower() == candidate.lower():
            logger.info(f"LLM matched (case‑insensitive): '{candidate}'")
            if query.lower() not in misspell_cache:
                misspell_cache[query.lower()] = candidate
                save_misspell_cache(misspell_cache)
            return candidate

    # Containment fallback
    for candidate in candidates:
        if answer and (
            answer.lower() in candidate.lower() or candidate.lower() in answer.lower()
        ):
            logger.info(f"LLM matched (containment): '{candidate}'")
            if query.lower() not in misspell_cache:
                misspell_cache[query.lower()] = candidate
                save_misspell_cache(misspell_cache)
            return candidate

    logger.info(f"LLM returned '{answer}' which is not in the candidate list.")
    # Fallback to best embedding match
    fallback = candidates[0]
    logger.info(f"Falling back to best embedding match: '{fallback}'")
    if query.lower() not in misspell_cache:
        misspell_cache[query.lower()] = fallback
        save_misspell_cache(misspell_cache)
    return fallback


# ------------------------------------------------------------
# Autocomplete endpoint (hybrid: prefix index + embeddings)
# ------------------------------------------------------------
@app.route("/autocomplete")
def autocomplete():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])

    now = datetime.now()

    # 1. Check in-memory time‑based cache
    if q in autocomplete_cache:
        titles, ts = autocomplete_cache[q]
        if now - ts < timedelta(seconds=CACHE_TTL_SECONDS):
            return jsonify(titles)

    # 2. Try prefix index (instant lookup)
    lower_q = q.lower()
    prefix_matches = prefix_index.get(lower_q, [])
    if len(prefix_matches) >= 5:
        results = prefix_matches[:10]
        autocomplete_cache[q] = (results, now)
        return jsonify(results)

    # 3. Fallback to embedding search for ambiguous queries
    top_titles = get_top_n_titles(q, n=10)
    autocomplete_cache[q] = (top_titles, now)

    # Limit cache size (keep last 1000 entries)
    if len(autocomplete_cache) > 1000:
        sorted_items = sorted(autocomplete_cache.items(), key=lambda x: x[1][1])
        for key, _ in sorted_items[:200]:
            del autocomplete_cache[key]

    return jsonify(top_titles)


# ------------------------------------------------------------
# Cosine similarity scoring for recommendations
# ------------------------------------------------------------
def get_similarity_score(index_a, index_b):
    vec_a = feature_matrix[index_a].reshape(1, -1)
    vec_b = feature_matrix[index_b].reshape(1, -1)
    score = cosine_similarity(vec_a, vec_b)[0][0]
    return round(float(score) * 100, 1)


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.form.get("query", "").strip()
    page = request.form.get("page", 1, type=int)

    if not user_input:
        return "Please enter a title."

    matched_title = match_title_with_llm(user_input, max_candidates=MAX_LLM_CANDIDATES)

    if matched_title is None:
        return render_template(
            "results.html",
            query=user_input,
            matched=None,
            recommendations=[],
            page=1,
            total_pages=1,
        )

    matched_index = title_to_index.get(matched_title)
    if matched_index is None:
        return "Error: matched title not found in dataset."

    matched_record = anime_data[matched_index]
    cluster_id = matched_record["cluster_id"]

    # Use pre‑built cluster index for speed
    candidate_indices = cluster_to_indices.get(cluster_id, [])
    scored_recommendations = []
    for idx in candidate_indices:
        if idx != matched_index:
            score = get_similarity_score(matched_index, idx)
            item_with_score = dict(anime_data[idx])
            item_with_score["similarity"] = score
            scored_recommendations.append(item_with_score)

    scored_recommendations.sort(key=lambda x: x["similarity"], reverse=True)

    total = len(scored_recommendations)
    total_pages = (total + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    page_results = scored_recommendations[start:end]

    return render_template(
        "results.html",
        query=user_input,
        matched=matched_record,
        recommendations=page_results,
        page=page,
        total_pages=total_pages,
    )


@app.route("/info")
def anime_info():
    title = request.args.get("title", "")
    if not title:
        return "No title provided.", 400
    idx = title_to_index.get(title)
    if idx is None:
        return "Anime not found.", 404
    anime = anime_data[idx]
    return render_template("info.html", anime=anime)


@app.route("/health")
def health():
    return {"status": "ok", "anime_count": len(anime_data)}


# ------------------------------------------------------------
# Start the Flask development server
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
