# Anime Recommender

A Flask web application that recommends anime titles using the Groq AI API for fuzzy/misspelled title matching.

## Architecture

- **Backend**: Python 3.12 + Flask
- **AI**: Groq API (`llama-3.1-8b-instant` model) for title matching
- **Data**: `anime_data.csv` — local CSV dataset with anime titles and cluster IDs
- **Templates**: Jinja2 HTML templates in `templates/`

## Key Files

- `app.py` — Main Flask application with routes and Groq integration
- `anime_data.csv` — Anime dataset with title and cluster_id columns
- `templates/index.html` — Single-page frontend with search form
- `requirements.txt` — Python dependencies

## Routes

- `GET /` — Home page with search form
- `POST /recommend` — Submit an anime title query, returns matched title
- `GET /test-match` — Test endpoint that tries to match "One Peace" → "One Piece"

## Environment Variables

- `GROQ_API_KEY` — Required. Groq API key for AI title matching (get one at https://console.groq.com)

## Running

The app runs via Flask dev server on `0.0.0.0:5000`.

For production, gunicorn is configured: `gunicorn --bind=0.0.0.0:5000 --reuse-port app:app`

## Deployment

Configured for autoscale deployment on Replit.
