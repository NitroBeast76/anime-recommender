# Import the Flask library to create a web application (a simple website that can respond to requests).
from flask import Flask

# Import pandas, a library for working with data (like reading CSV files and handling tables).
import pandas as pd

# Import os to interact with the operating system (e.g., to read environment variables like API keys).
import os

# Import the Groq library to communicate with Groq's AI models (like Llama or Mixtral) for text generation.
from groq import Groq

# Import load_dotenv from the dotenv package to load environment variables from a .env file.
# This keeps sensitive information (like API keys) out of the source code.
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists. This sets them as system environment variables.
load_dotenv()

# Initialize the Groq client with our API key, which we get from the environment variable "GROQ_API_KEY".
# This client object will handle all requests to Groq's AI services.
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Choose the AI model to use. Groq offers various models; here we pick "llama-3.1-8b-instant" which is fast and free.
# You could also use a larger model like "mixtral-8x7b-32768" if you need more power. # or "mixtral-8x7b-32768"
MODEL_NAME = "llama-3.1-8b-instant"  # – updated from decommissioned llama3-8b-8192

# Create a Flask web application instance. This will be our web server.
app = Flask(__name__)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
def load_anime_data(filename="anime_data.csv"):
    """
    Reads the CSV file containing anime data and returns it as a list of dictionaries.
    Each dictionary represents one anime row, with column names as keys.
    The cluster_id column is converted to integer for consistency.
    """
    # Read the CSV file into a pandas DataFrame (a table-like structure).
    df = pd.read_csv(filename, encoding="utf-8")

    # Ensure the 'cluster_id' column is of integer type (sometimes it might be read as float).
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Convert the DataFrame to a list of dictionaries. Each dictionary corresponds to one row,
    # with keys being the column names and values being the cell values.
    data = df.to_dict(orient="records")
    return data


# Load the anime data from the CSV file when the script starts, so it's ready for use later.
# The data is stored in the global variable `anime_data`.
anime_data = load_anime_data()


# ------------------------------------------------------------
# Title matching function using Groq
# ------------------------------------------------------------
def match_title_with_groq(user_input, title_list):
    """
    Uses Groq's large language model to find the closest matching title from a given list.
    The user input can be misspelled or incomplete. The function returns the exact matched title
    from the list, or None if no good match is found.
    """
    # Convert the list of titles into a bulleted string, one title per line starting with "- ".
    # This will be inserted into the prompt to help the AI understand the available options.
    titles_formatted = "\n- ".join(title_list)

    # Build the prompt that will be sent to the AI.
    # The prompt explains the task, provides the list of valid titles, and gives the user query.
    # We instruct the AI to return only the exact title or "NOT_FOUND".
    prompt = f"""You are a title‑matching assistant. Given a user query (which may be misspelled or incomplete), find the closest matching title from the following list. Return ONLY the exact title as it appears in the list. If no reasonable match exists, return exactly "NOT_FOUND".

List of valid titles:
- {titles_formatted}

User query: {user_input}
"""
    try:
        # Call the Groq API to generate a completion based on our prompt.
        chat_completion = client.chat.completions.create(
            messages=[
                # System message sets the behavior of the assistant.
                {
                    "role": "system",
                    "content": "You are a helpful assistant that returns only the matched title or NOT_FOUND.",
                },
                # User message contains our prompt with the titles and the user query.
                {"role": "user", "content": prompt},
            ],
            model=MODEL_NAME,  # Use the model we selected earlier.
            temperature=0.0,  # Set temperature to 0 for deterministic, repeatable output (no creativity).
            max_tokens=50,  # Limit the response to 50 tokens, enough for a title.
        )

        # Extract the content of the assistant's reply and strip any extra whitespace.
        result = chat_completion.choices[0].message.content.strip()

        # If the AI returned "NOT_FOUND", that means no good match was found.
        if result == "NOT_FOUND":
            return None

        # Optional but good practice: verify that the returned title actually exists in our list.
        # This guards against the AI occasionally hallucinating a title not in the list.
        if result in title_list:
            return result
        else:
            # If the AI returned something not in the list, treat it as no match.
            return None
    except Exception as e:
        # If any error occurs (network issue, API error, etc.), print it and return None.
        print(f"Groq API error: {e}")
        return None


# ------------------------------------------------------------
# Routes (URL endpoints)
# ------------------------------------------------------------
# Define the route for the root URL "/". When someone visits the homepage, this function runs.
@app.route("/")
def home():
    # Return a simple plain text greeting.
    return "Hello, World!"


# This is a test route – you can remove it later once you build the real recommendation endpoint.
# When you visit "/test-match" in your browser, it will attempt to match a sample query.
@app.route("/test-match")
def test_match():
    # A sample query that might be misspelled (here "One Peace" instead of "One Piece").
    query = "One Peace"  # You can change this or pass it as a parameter later.

    # Extract all anime titles from our loaded data (each item has a 'title' field).
    titles = [item["title"] for item in anime_data]

    # Call our matching function to find the closest title.
    matched = match_title_with_groq(query, titles)

    # Return a response showing what was matched.
    if matched:
        return f"Matched: {matched}"
    else:
        return "No match found"


# This block ensures that the Flask development server runs only if this script is executed directly,
# not if it is imported as a module by another script.
if __name__ == "__main__":
    # Start the Flask development server with debug mode enabled.
    # Debug mode automatically restarts the server on code changes and shows detailed error pages.
    app.run(debug=True)
