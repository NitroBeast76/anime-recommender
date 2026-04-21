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
    """
    print("Loading anime dataset...")
    anime_df = pd.read_csv(anime_file, encoding="utf-8")
    anime_df["cluster_id"] = anime_df["cluster_id"].astype(int)

    print("Loading feature matrix...")
    feature_df = pd.read_csv(feature_file, encoding="utf-8")
    if "cluster_id" in feature_df.columns:
        feature_df = feature_df.drop(columns=["cluster_id"])

    # Sort both alphabetically
    anime_df = anime_df.sort_values(by="title").reset_index(drop=True)

    if "title" in feature_df.columns:
        feature_df = feature_df.sort_values(by="title").reset_index(drop=True)
        mismatches = (anime_df["title"].values != feature_df["title"].values).sum()
        if mismatches > 0:
            raise ValueError(
                f"ALIGNMENT ERROR: {mismatches} rows do not match between "
                f"{anime_file} and {feature_file}."
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
# Build or load combined title embeddings
# ------------------------------------------------------------
def load_title_embeddings(anime_data, cache_file="title_embeddings.pkl"):
    """
    Embeds a combined text of title + english_title + japanese_title for each anime.
    Gracefully handles missing columns (e.g., in dummy data).
    """
    if os.path.exists(cache_file):
        print("Loading cached title embeddings...")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        current_titles = [item["title"] for item in anime_data]
        if cache.get("titles") == current_titles:
            print(f"Embeddings loaded from cache ({len(cache['titles'])} titles).")
            return cache["titles"], cache["embeddings"]
        else:
            print("Dataset has changed — rebuilding embeddings cache...")

    print("Building combined title embeddings...")
    combined_texts = []
    for item in anime_data:
        parts = [item["title"]]
        # Check if alternate title columns exist and are not "Unknown"
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

    print(f"Embeddings built and cached ({len(titles)} titles).")
    return titles, embeddings


title_list, title_embeddings = load_title_embeddings(anime_data)


# ------------------------------------------------------------
# Title matching using embeddings (single‑stage)
# ------------------------------------------------------------
def match_title_with_embeddings(user_input, confidence_threshold=0.5):
    """
    Embeds the user query and finds the closest title in the precomputed
    combined embeddings. Returns None if confidence is below threshold.
    """
    query_embedding = embedding_model.encode([user_input], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, title_embeddings)[0]

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    best_title = title_list[best_index]

    print(f"Query: '{user_input}'")
    print(f"Best match: '{best_title}' (score: {best_score:.3f})")

    if best_score < confidence_threshold:
        print(f"Score too low — no confident match found.")
        return None

    return best_title


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

    # Find matched record and index
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

    scored_recommendations = []
    for i, item in enumerate(anime_data):
        if item["cluster_id"] == cluster_id and item["title"] != matched_title:
            score = get_similarity_score(matched_index, i)
            item_with_score = dict(item)
            item_with_score["similarity"] = score
            scored_recommendations.append(item_with_score)

    scored_recommendations.sort(key=lambda x: x["similarity"], reverse=True)

    total = len(scored_recommendations)
    total_pages = max(1, -(-total // RESULTS_PER_PAGE))
    page = max(1, min(page, total_pages))
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

    anime = next((item for item in anime_data if item["title"] == title), None)
    if anime is None:
        return "Anime not found.", 404

    return render_template("info.html", anime=anime)


# ------------------------------------------------------------
# Start the Flask development server
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
