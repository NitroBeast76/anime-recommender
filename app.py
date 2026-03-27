# ------------------------------------------------------------
# Import necessary libraries
# ------------------------------------------------------------
# Flask: a lightweight web framework for Python. It helps us create a web application
# that can handle requests from users (like visiting a page or submitting a form).
from flask import Flask, render_template, request

from urllib.parse import unquote  # add at top if not already

# Pandas: a library for data manipulation and analysis. We'll use it to read the CSV file
# containing anime information and work with it as a table (DataFrame).
import pandas as pd

# OS module: provides functions to interact with the operating system.
# We'll use it to read environment variables (like our API key) securely.
import os

# Groq library: gives us a simple way to send requests to Groq's AI models (like Llama).
# This allows us to use AI for tasks like matching user input to anime titles.
from groq import Groq

# dotenv: a library that loads environment variables from a .env file into the system.
# This keeps sensitive information (like API keys) out of our source code.
from dotenv import load_dotenv

# Simple cache for matched titles to avoid repeated API calls
match_cache = {}

# ------------------------------------------------------------
# Load environment variables and set up Groq client
# ------------------------------------------------------------
# Load any variables defined in a .env file (if present) so they become available
# as environment variables. This is commonly done to keep API keys secure.
load_dotenv()

# Create a Groq client object. This client will handle all communication with Groq's API.
# We pass it our API key, which we retrieve from the environment variable "GROQ_API_KEY".
# os.getenv("GROQ_API_KEY") looks for that variable and returns its value.
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Specify which AI model we want to use. Groq offers several models; here we choose
# "llama-3.1-8b-instant", a fast and free model suitable for many tasks.
# You could change this to a larger model like "mixtral-8x7b-32768" for more complex tasks.
MODEL_NAME = "llama-3.1-8b-instant"  # Updated from an older model name

# ------------------------------------------------------------
# Initialize the Flask application
# ------------------------------------------------------------
# Create an instance of the Flask class. This object will be our web server.
# The name __name__ helps Flask locate resources like templates and static files.
app = Flask(__name__)


# ------------------------------------------------------------
# Load the anime dataset from a CSV file
# ------------------------------------------------------------
def load_anime_data(filename="anime_data.csv"):
    df = pd.read_csv(filename, encoding="utf-8")
    df["cluster_id"] = df["cluster_id"].astype(int)
    # No longer need to add image_url – it's already in the CSV
    data = df.to_dict(orient="records")
    return data


# Load the anime data once when the script starts, so it's ready for use later.
# The data is stored in a global variable `anime_data` so all functions can access it.
anime_data = load_anime_data()


# ------------------------------------------------------------
# Title matching function using Groq AI
# ------------------------------------------------------------
def match_title_with_groq(user_input, title_list):
            # Normalize input for cache key (lowercase, strip whitespace)
            cache_key = user_input.strip().lower()

            # Check cache first
            if cache_key in match_cache:
                print(f"Cache hit for '{user_input}' -> {match_cache[cache_key]}")
                return match_cache[cache_key]

            titles_formatted = "\n- ".join(title_list)
            prompt = f"""You are a title‑matching assistant. Given a user query (which may be misspelled or incomplete), find the closest matching title from the following list. Return ONLY the exact title as it appears in the list. If no reasonable match exists, return exactly "NOT_FOUND".

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
                print(f"Query: '{user_input}'")
                print(f"Raw result: '{result}'")

                if result == "NOT_FOUND":
                    match_cache[cache_key] = None
                    return None
                if result in title_list:
                    match_cache[cache_key] = result
                    return result
                else:
                    print(f"Warning: LLM returned '{result}' which is not in the title list.")
                    match_cache[cache_key] = None
                    return None
            except Exception as e:
                print(f"Groq API error: {type(e).__name__}: {e}")
                # Don't cache errors – next time it will try again
                return None


# ------------------------------------------------------------
# Define the web routes (URLs) that our application will respond to
# ------------------------------------------------------------
# Route for the homepage ("/")
@app.route("/")
def home():
    """
    Handles requests to the root URL (e.g., http://localhost:5000/).
    Renders and returns the main HTML page (index.html) where users can enter an anime title.
    """
    return render_template("index.html")


# A test route to experiment with the title matching function.
# You can visit "/test-match" in your browser to see it in action.
@app.route("/test-match")
def test_match():
    """
    A simple test endpoint that tries to match a sample query ("One Peace") against the anime list.
    Useful for checking if the Groq integration is working without building a full form.
    """
    # Sample query that might be misspelled (e.g., "One Peace" instead of "One Piece").
    query = "One Peace"  # You can change this or pass it as a URL parameter later.

    # Extract all anime titles from our loaded data into a list.
    titles = [item["title"] for item in anime_data]

    # Call our matching function to find the closest title.
    matched = match_title_with_groq(query, titles)

    # Return a simple text response showing what was matched.
    if matched:
        return f"Did you mean: {matched}?"
    else:
        return "No match found."


# Route for receiving the user's query and returning recommendations.
# This route only responds to POST requests (when the user submits the form).
@app.route("/recommend", methods=["POST"])
def recommend():
        user_input = request.form.get("query", "").strip()
        if not user_input:
            return "Please enter a title."

        titles = [item["title"] for item in anime_data]
        matched_title = match_title_with_groq(user_input, titles)

        if matched_title is None:
            return render_template("results.html", 
                                   query=user_input, 
                                   matched=None, 
                                   recommendations=[])

        # Find the matched anime record
        matched_record = None
        for item in anime_data:
            if item["title"] == matched_title:
                matched_record = item
                break

        if matched_record is None:
            return "Error: matched title not found in dataset."

        cluster_id = matched_record["cluster_id"]

        # Gather all records in the same cluster (excluding the matched one)
        recommendations = [
            item for item in anime_data
            if item["cluster_id"] == cluster_id and item["title"] != matched_title
        ]

        return render_template("results.html", 
                               query=user_input, 
                               matched=matched_record, 
                               recommendations=recommendations)


@app.route("/info")
def anime_info():
    title = request.args.get("title", "")
    if not title:
        return "No title provided.", 400

    # Find the anime record by exact title
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
# This block ensures that the server runs only when this script is executed directly,
# not when it is imported as a module by another script.
if __name__ == "__main__":
    # Run the app on all available network interfaces (host="0.0.0.0") so it can be accessed
    # from other devices on the same network, and on port 5000.
    # The debug=True option enables automatic reloading when code changes and shows detailed errors.
    # In production, debug should be set to False.
    app.run(host="0.0.0.0", port=5000, debug=True)
