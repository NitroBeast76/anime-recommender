# ------------------------------------------------------------
# Import necessary libraries
# ------------------------------------------------------------
# Flask: a lightweight web framework for Python. It helps us create a web application
# that can handle requests from users (like visiting a page or submitting a form).
from flask import Flask, render_template, request

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
    """
    Reads a CSV file containing anime information and returns it as a list of dictionaries.

    Each dictionary represents one anime row, with column names as keys.
    The 'cluster_id' column is converted to an integer for consistency (sometimes it might be read as a float).
    """
    # Use pandas to read the CSV file into a DataFrame (a table-like structure).
    # We specify encoding='utf-8' to handle special characters properly.
    df = pd.read_csv(filename, encoding="utf-8")

    # Convert the 'cluster_id' column to integer type. This ensures that cluster IDs
    # are whole numbers, which is important for grouping later.
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Convert the DataFrame to a list of dictionaries. Each dictionary's keys are the column names,
    # and the values are the corresponding data for that row.
    # 'orient="records"' makes a list of dictionaries, one per row.
    data = df.to_dict(orient="records")
    return data


# Load the anime data once when the script starts, so it's ready for use later.
# The data is stored in a global variable `anime_data` so all functions can access it.
anime_data = load_anime_data()


# ------------------------------------------------------------
# Title matching function using Groq AI
# ------------------------------------------------------------
def match_title_with_groq(user_input, title_list):
    """
    Uses Groq's AI to find the closest matching anime title from a given list,
    even if the user's input is misspelled or incomplete.

    Parameters:
        user_input (str): The text entered by the user.
        title_list (list): A list of valid anime titles from our dataset.

    Returns:
        str or None: The matched title exactly as it appears in the list, or None if no good match is found.
    """
    # Format the list of titles into a string with each title on a new line preceded by a dash.
    # This makes it easy for the AI to see all available options.
    titles_formatted = "\n- ".join(title_list)

    # Construct a prompt that clearly instructs the AI what to do.
    # We ask it to return only the exact title from the list, or "NOT_FOUND" if there's no reasonable match.
    prompt = f"""You are a title‑matching assistant. Given a user query (which may be misspelled or incomplete), find the closest matching title from the following list. Return ONLY the exact title as it appears in the list. If no reasonable match exists, return exactly "NOT_FOUND".

List of valid titles:
- {titles_formatted}

User query: {user_input}
"""
    try:
        # Send a request to Groq's chat completion API.
        # We provide a system message to set the AI's behavior, and the user message with our prompt.
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that returns only the matched title or NOT_FOUND.",
                },
                {"role": "user", "content": prompt},
            ],
            model=MODEL_NAME,  # Use the model we chose earlier
            temperature=0.0,  # Set temperature to 0 for deterministic (repeatable) output
            max_tokens=50,  # Limit the response to 50 tokens (enough for a title)
        )

        # Extract the content of the AI's response and remove any extra whitespace.
        result = chat_completion.choices[0].message.content.strip()

        # For debugging, print the query and the raw result to the console.
        print(f"Query: '{user_input}'")
        print(f"Raw result: '{result}'")

        # If the AI returned "NOT_FOUND", it means no good match exists.
        if result == "NOT_FOUND":
            return None

        # If the result is actually one of the titles in our list, return it.
        if result in title_list:
            return result
        else:
            # If the AI returned something that isn't in the list (shouldn't happen, but just in case),
            # print a warning and return None.
            print(f"Warning: '{result}' not in list")
            return None
    except Exception as e:
        # If any error occurs during the API call (network issue, invalid key, etc.),
        # print the error and return None so the application doesn't crash.
        print(f"Groq API error: {type(e).__name__}: {e}")
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
    """
    Handles form submissions from the homepage. Extracts the user's input,
    matches it to a known anime title using Groq, finds other anime in the same cluster,
    and displays the results.
    """
    # Get the value of the form field named "query". If not present, default to empty string.
    user_input = request.form.get("query", "").strip()

    # If the user didn't enter anything, show a simple error message.
    if not user_input:
        return "Please enter a title."

    # Extract all titles from the dataset.
    titles = [item["title"] for item in anime_data]

    # Use Groq to find the best matching title.
    matched_title = match_title_with_groq(user_input, titles)

    # If no match was found, render the results page with a "Not found" message and no recommendations.
    if matched_title is None:
        return render_template(
            "results.html", query=user_input, matched="Not found", recommendations=[]
        )

    # Find the cluster ID of the matched title by searching through our data.
    cluster_id = None
    for item in anime_data:
        if item["title"] == matched_title:
            cluster_id = item["cluster_id"]
            break

    # Prepare a list of recommendations: all anime titles in the same cluster,
    # excluding the matched title itself.
    recommendations = []
    if cluster_id is not None:
        recommendations = [
            item["title"]
            for item in anime_data
            if item["cluster_id"] == cluster_id and item["title"] != matched_title
        ]

    # Render the results page, passing the original query, the matched title, and the recommendations.
    return render_template(
        "results.html",
        query=user_input,
        matched=matched_title,
        recommendations=recommendations,
    )


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
