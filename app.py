# ------------------------------------------------------------
# Import necessary libraries
# ------------------------------------------------------------
from flask import Flask, render_template, request
from urllib.parse import unquote

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import os
import pickle
from dotenv import load_dotenv

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
# Load the embedding model
# ------------------------------------------------------------
# This model runs locally on your computer — no API key needed.
# It downloads once the first time (~90MB) and is cached after that.
# "all-MiniLM-L6-v2" is small, fast, and very good at matching text.
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

# ------------------------------------------------------------
# Load and verify anime data and feature matrix
# ------------------------------------------------------------
def load_and_verify(
    anime_file="anime_data.csv", feature_file="anime_features_scaled.csv"
):
    """
    Loads both the anime dataset and the feature matrix, sorts both
    alphabetically by title, and verifies they are aligned row by row.
    If they are out of sync the app stops immediately and tells you why.
    """
    print("Loading anime dataset...")
    anime_df = pd.read_csv(anime_file, encoding="utf-8")
    anime_df["cluster_id"] = anime_df["cluster_id"].astype(int)

    print("Loading feature matrix...")
    feature_df = pd.read_csv(feature_file, encoding="utf-8")
    if "cluster_id" in feature_df.columns:
        feature_df = feature_df.drop(columns=["cluster_id"])

    # Sort both alphabetically so row order is always guaranteed to match
    anime_df = anime_df.sort_values(by="title").reset_index(drop=True)

    if "title" in feature_df.columns:
        feature_df = feature_df.sort_values(by="title").reset_index(drop=True)

        mismatches = (anime_df["title"].values != feature_df["title"].values).sum()

        if mismatches > 0:
            raise ValueError(
                f"ALIGNMENT ERROR: {mismatches} rows do not match between "
                f"{anime_file} and {feature_file}. "
                f"Check that both files contain the same titles."
            )
        else:
            print("Alignment check passed — both files are in the same order.")

        feature_df = feature_df.drop(columns=["title"])
    else:
        print("Warning: feature matrix has no title column — cannot verify alignment.")
        print("Assuming both files are in alphabetical order by title.")

    anime_data = anime_df.to_dict(orient="records")
    feature_matrix = feature_df.values

    print(f"Loaded {len(anime_data)} anime titles.")
    print(f"Feature matrix shape: {feature_matrix.shape}")

    return anime_data, feature_matrix


anime_data, feature_matrix = load_and_verify()


# ------------------------------------------------------------
# Build or load the title embeddings
# ------------------------------------------------------------
def load_title_embeddings(anime_data, cache_file="title_embeddings.pkl"):
    """
    Embeds every anime title in the dataset and stores the result.

    The first time this runs it will take a minute or two to embed
    all 28,000 titles. After that it saves the result to a .pkl file
    so every future startup just loads that file instantly instead.

    How it works:
    - Each title becomes a list of 384 numbers that represents its meaning.
    - Similar titles (like "One Piece" and "One Piece!") get similar numbers.
    - When a user searches, we embed their query the same way and find
      whichever stored title has the closest numbers — that's the match.
    """

    if os.path.exists(cache_file):
        # Load the pre-computed embeddings from disk
        print("Loading cached title embeddings...")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)

        # Make sure the cache still matches the current dataset
        # If titles have changed since the cache was built, rebuild it
        if cache["titles"] == [item["title"] for item in anime_data]:
            print(f"Embeddings loaded from cache ({len(cache['titles'])} titles).")
            return cache["titles"], cache["embeddings"]
        else:
            print("Dataset has changed — rebuilding embeddings cache...")

    # No cache found or cache is stale — embed all titles from scratch
    print("Building title embeddings for the first time...")
    print("This will take 1-2 minutes for a large dataset. Please wait.")

    titles = [item["title"] for item in anime_data]

    # embed all titles in one batch — much faster than one at a time
    embeddings = embedding_model.encode(
        titles,
        batch_size=64,        # Process 64 titles at a time
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Save to cache so next startup is instant
    with open(cache_file, "wb") as f:
        pickle.dump({"titles": titles, "embeddings": embeddings}, f)

    print(f"Embeddings built and cached ({len(titles)} titles).")
    return titles, embeddings


title_list, title_embeddings = load_title_embeddings(anime_data)


# ------------------------------------------------------------
# Title matching using embeddings
# ------------------------------------------------------------
def match_title_with_embeddings(user_input):
    """
    Finds the closest anime title to whatever the user typed.

    Instead of sending 28,000 titles to an AI model (which hits token limits),
    this embeds just the user's query — one tiny local operation — and then
    finds which stored title embedding is mathematically closest to it.

    This handles typos, partial titles, and alternate spellings naturally
    because similar-sounding text produces similar embeddings.

    Returns the best matching title string, or None if the best match
    score is too low to be considered a real match.
    """
    # Embed just the user's query — this is fast and local, no API call
    query_embedding = embedding_model.encode(
        [user_input],
        convert_to_numpy=True
    )

    # Compare the query embedding against every stored title embedding
    # This gives a similarity score between 0 and 1 for each title
    scores = cosine_similarity(query_embedding, title_embeddings)[0]

    # Find the index of the highest scoring title
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    best_title = title_list[best_index]

    print(f"Query: '{user_input}'")
    print(f"Best match: '{best_title}' (score: {best_score:.3f})")

    # If the best score is below this threshold, treat it as no match found.
    # 0.5 means the query and title share at least moderate semantic similarity.
    # You can raise this (e.g. 0.6) to be stricter, or lower it to be more lenient.
    MATCH_THRESHOLD = 0.5

    if best_score < MATCH_THRESHOLD:
        print(f"Score too low — no confident match found.")
        return None

    return best_title


# ------------------------------------------------------------
# Cosine similarity scoring for recommendations
# ------------------------------------------------------------
def get_similarity_score(index_a, index_b):
    """
    Calculates how similar two anime are using their feature vectors.
    Returns a percentage between 0 and 100.
    """
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
    """
    Main recommendation route.

    1. Get the title the user typed.
    2. Use embeddings to find the closest real title — no token limits.
    3. Find that title's cluster and row index.
    4. Score all other anime in the same cluster by cosine similarity.
    5. Sort highest first, paginate, send to the results page.
    """
    user_input = request.form.get("query", "").strip()
    page = request.form.get("page", 1, type=int)

    if not user_input:
        return "Please enter a title."

    # Match the user's query using embeddings instead of the LLM
    matched_title = match_title_with_embeddings(user_input)

    if matched_title is None:
        return render_template(
            "results.html",
            query=user_input,
            matched=None,
            recommendations=[],
            page=1,
            total_pages=1,
        )

    # Find the matched anime record and its row index
    matched_record = None
    matched_index = None
    for i, item in enumerate(anime_data):
        if item["title"] == matched_title:
            matched_record = item
            matched_index = i
            break

    if matched_record is None:
        return "Error: matched title not found in dataset."

    cluster_id = matched_record["cluster_id"]

    # Score every other anime in the same cluster
    scored_recommendations = []
    for i, item in enumerate(anime_data):
        if item["cluster_id"] == cluster_id and item["title"] != matched_title:
            score = get_similarity_score(matched_index, i)
            item_with_score = dict(item)
            item_with_score["similarity"] = score
            scored_recommendations.append(item_with_score)

    # Sort by similarity, highest first
    scored_recommendations.sort(key=lambda x: x["similarity"], reverse=True)

    # Paginate
    total = len(scored_recommendations)
    total_pages = max(1, -(-total // RESULTS_PER_PAGE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    page_results = scored_recommendations[start:end]

    print(f"Returning page {page} of {total_pages} ({len(page_results)} results)")

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

    anime = None
    for item in anime_data:
        if item["title"] == title:
            anime = item
            break

    if anime is None:
        return "Anime not found.", 404

    return render_template("info.html", anime=anime)


# ------------------------------------------------------------
# Start the Flask development server
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)