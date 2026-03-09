# Anime Recommender

A simple web app that takes a user‑typed anime title (even misspelled) and returns recommendations from the same cluster, using Groq's LLM to match titles.

## Setup

1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file and add your Groq API key:  
   `GROQ_API_KEY=your_key_here`
6. Run the app: `python app.py`
7. Visit `http://127.0.0.1:5000`

## How it works

- The homepage has a search box (you'll build that next).
- The `/test-match` route demonstrates title matching.
- Uses Groq's `llama-3.1-8b-instant` model (free tier).