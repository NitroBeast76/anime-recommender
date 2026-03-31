# ------------------------------------------------------------
# Import necessary libraries
# ------------------------------------------------------------
from flask import Flask, render_template, request
from urllib.parse import unquote

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import os
from groq import Groq
from dotenv import load_dotenv

# Simple cache for matched titles to avoid repeated API calls
match_cache = {}

# ------------------------------------------------------------
# Load environment variables and set up Groq client
# ------------------------------------------------------------
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"

# ------------------------------------------------------------
# Initialize the Flask application
# ------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------
# How many recommendations to show per page
# ------------------------------------------------------------
# Change this number to show more or fewer results per page.
# For example, set to 20 to show 20 cards per page.
RESULTS_PER_PAGE = 20

# ------------------------------------------------------------
# Load the anime dataset and the feature matrix
# ------------------------------------------------------------
def load_anime_data(filename="anime_data.csv"):
    """
    Loads the main anime dataset — the human-readable one with titles,
    synopsis, images etc. This is what gets displayed on the website.
    """
    df = pd.read_csv(filename, encoding="utf-8")
    df["cluster_id"] = df["cluster_id"].astype(int)
    data = df.to_dict(orient="records")
    return data


def load_feature_matrix(filename="anime_features_scaled.csv"):
    """
    Loads the feature matrix produced by Category 1.
    This is the numerical version of the dataset — all 0s, 1s, and scaled numbers.
    We use this to calculate how similar two anime are to each other.

    IMPORTANT: The rows in this file must be in the exact same order
    as the rows in anime_data.csv. If they ever get out of sync,
    the similarity scores will be wrong.
    """
    df = pd.read_csv(filename, encoding="utf-8")

    # Drop the cluster_id column if it's in here — we only want the features
    if "cluster_id" in df.columns:
        df = df.drop(columns=["cluster_id"])

    return df.values  # Return as a numpy array for fast calculations


# Load everything once when the website starts
anime_data = load_anime_data()
feature_matrix = load_feature_matrix()

print(f"Loaded {len(anime_data)} anime titles.")
print(f"Feature matrix shape: {feature_matrix.shape}")


# ------------------------------------------------------------
# Title matching function using Groq AI
# ------------------------------------------------------------
def match_title_with_groq(user_input, title_list):
    """
    Takes whatever the user typed (even misspelled) and finds the closest
    real title from our dataset using an AI model.
    Results are cached so we don't call the API twice for the same query.
    """
    cache_key = user_input.strip().lower()

    if cache_key in match_cache:
        print(f"Cache hit for '{user_input}' -> {match_cache[cache_key]}")
        return match_cache[cache_key]

    titles_formatted = "\n- ".join(title_list)
    prompt = f"""You are a title-matching assistant. Given a user query (which may be misspelled or incomplete), find the closest matching title from the following list. Return ONLY the exact title as it appears in the list. If no reasonable match exists, return exactly "NOT_FOUND".

List of valid titles:
- {titles_formatted}

User query: {user_input}
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns only the matched title or NOT_FOUND."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=50
        )
        result = chat_completion.choices[0].message.content.strip()
        print(f"Query: '{user_input}' → Matched: '{result}'")

        if result == "NOT_FOUND":
            match_cache[cache_key] = None
            return None

        if result in title_list:
            match_cache[cache_key] = result
            return result
        else:
            print(f"Warning: AI returned '{result}' which is not in the title list.")
            match_cache[cache_key] = None
            return None

    except Exception as e:
        print(f"Groq API error: {type(e).__name__}: {e}")
        return None


# ------------------------------------------------------------
# Cosine similarity scoring
# ------------------------------------------------------------
def get_similarity_score(index_a, index_b):
    """
    Calculates how similar two anime are to each other using their
    feature vectors from the feature matrix.

    Returns a percentage between 0 and 100.
    100 means identical. 0 means nothing in common.

    How it works:
    - Each anime is represented as a row of numbers in the feature matrix.
    - Cosine similarity measures the angle between those two rows.
    - A small angle = very similar. A large angle = very different.
    - We multiply by 100 to turn it into a percentage.
    """
    vec_a = feature_matrix[index_a].reshape(1, -1)
    vec_b = feature_matrix[index_b].reshape(1, -1)
    score = cosine_similarity(vec_a, vec_b)[0][0]

    # Round to one decimal place for clean display (e.g. 87.3%)
    return round(float(score) * 100, 1)


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test-match")
def test_match():
    query = "One Peace"
    titles = [item["title"] for item in anime_data]
    matched = match_title_with_groq(query, titles)
    if matched:
        return f"Did you mean: {matched}?"
    else:
        return "No match found."


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Main recommendation route.

    Steps:
    1. Get the title the user typed.
    2. Use AI to find the closest real title in our dataset.
    3. Find that title's cluster ID and its position in the feature matrix.
    4. Find all other anime in the same cluster.
    5. Calculate a cosine similarity percentage for each one vs the searched title.
    6. Sort by similarity score, highest first.
    7. Paginate the results so we don't show hundreds of cards at once.
    8. Send everything to the results page.
    """
    user_input = request.form.get("query", "").strip()
    page = request.form.get("page", 1, type=int)

    if not user_input:
        return "Please enter a title."

    titles = [item["title"] for item in anime_data]
    matched_title = match_title_with_groq(user_input, titles)

    if matched_title is None:
        return render_template("results.html",
                               query=user_input,
                               matched=None,
                               recommendations=[],
                               page=1,
                               total_pages=1)

    # Find the matched anime and its index in the dataset
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

    # Find all other anime in the same cluster and score each one
    scored_recommendations = []
    for i, item in enumerate(anime_data):
        if item["cluster_id"] == cluster_id and item["title"] != matched_title:
            score = get_similarity_score(matched_index, i)
            # Add the score to a copy of the item so the template can display it
            item_with_score = dict(item)
            item_with_score["similarity"] = score
            scored_recommendations.append(item_with_score)

    # Sort by similarity score, highest first
    scored_recommendations.sort(key=lambda x: x["similarity"], reverse=True)

    # Paginate — split the full list into pages of RESULTS_PER_PAGE
    total = len(scored_recommendations)
    total_pages = max(1, -(-total // RESULTS_PER_PAGE))  # Ceiling division
    page = max(1, min(page, total_pages))                 # Clamp to valid range

    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    page_results = scored_recommendations[start:end]

    print(f"Returning page {page} of {total_pages} ({len(page_results)} results)")

    return render_template("results.html",
                           query=user_input,
                           matched=matched_record,
                           recommendations=page_results,
                           page=page,
                           total_pages=total_pages)


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