# Anime Dataset Schema — Final Version

This document describes the structure and encoding logic of the cleaned
anime catalogue (`anime_clean.csv`) and the derived feature matrix
(`anime_features_scaled.csv`). It is intended for data scientists,
engineers, and any downstream consumers of these files.

---

## 1. Source Data Cleansing

### Content Safety
- All titles tagged with the genres **Hentai**, **Erotica**, or **Ecchi**
  have been removed from the working dataset (see `anime_adult.csv`).
- Entries rated **R+ - Mild Nudity** that lacked an explicit adult genre
  were also treated as adult content and quarantined.
- Promotional videos (`type == "PV"`), cluster‑0 noise entries, and
  titles marked “Currently Airing” with a year more than two years in the
  past have been excluded.

### Missing Data Handling
- **Year**: Missing or unrecoverable years are stored as **`-1`** (a
  sentinel value). A boolean column `year_is_imputed` indicates which
  rows are missing.
- **Episodes**: Missing episodes are filled with **1**; a boolean
  `episodes_is_imputed` tracks these rows.
- **Synopsis**: Rows with fewer than 30 characters of synopsis text are
  flagged with `synopsis_stub = True`.
- **Images**: Broken placeholder URLs from `via.placeholder.com` have been
  replaced with a self‑hosted fallback (`/static/no_image.png`).
- **English titles**: Missing English titles are left as NULL; the primary
  display fallback is the romanised `title` column.
- **Source**: “Unknown” means the adaptation source information was not
  available on MyAnimeList and is truly unknown.

---

## 2. Feature Matrix (`anime_features_scaled.csv`)

### 2.1 Categorical One‑Hot Groups
- **Genres** (21 columns), **Themes** (52 columns), **Demographics** (5 columns):
  binary 0/1. Adult genre columns (Hentai, Erotica, Ecchi) are excluded.
- **Type** (TV, Movie, OVA, etc.), **Source** (Manga, Original, etc.),
  **Rating** (PG, PG‑13, R, etc.), **Duration Bucket** (Short, Standard, Long),
  **Decade** (1910s–2020s, Unknown): one‑hot encoded.

### 2.2 Studio Encoding
- The full list of 1,104 studios has been reduced to the **Top‑50 most
  frequent** studios plus an **“Other”** catch‑all category.
- Each row is **L2‑normalised** (unit norm) within the studio group.
  For an anime produced by N studios, each studio column receives a value
  of **1/√N**. This prevents multi‑studio productions from dominating
  distance calculations.

### 2.3 Year Scaling
- Only **non‑imputed** years are min‑max scaled to `[0, 1]`.
- Rows where `year_is_imputed == True` receive a **neutral value of 0.5**,
  regardless of the sentinel `-1` in the raw data.
- The `year_is_imputed` boolean is included as a separate feature.

### 2.4 Episodes Scaling
- The raw episode count is transformed with **log1p** (logarithm of
  1 + episodes) to reduce extreme skew, then min‑max scaled to `[0, 1]`.
- Rows with `episodes_is_imputed == True` are assigned the **mean** of
  the scaled non‑imputed values.
- The `episodes_is_imputed` flag is included as a separate feature.

### 2.5 Synopsis Embeddings
- Generated with the **Sentence‑Transformer model `all-MiniLM-L6-v2`**.
- Synopsis text shorter than 30 characters (stubs) is replaced with an
  empty string before encoding, so those rows receive near‑zero embeddings
  rather than garbage vectors.
- The resulting 384‑dimensional embeddings are reduced to **50 principal
  components** via PCA (retaining the maximum meaningful variance).
- Column names: `syn_emb_1` through `syn_emb_50`.

### 2.6 Synopsis Sentiment
- Computed with **VADER** (Valence Aware Dictionary and sEntiment Reasoner).
- When the synopsis is a stub (`synopsis_stub = True`), the sentiment
  value would naïvely be 0.0 (neutral). To prevent these rows from
  clustering artificially, stub rows are assigned the **global mean
  sentiment** of all non‑stub entries.
- A boolean `sentiment_is_default` indicates rows that received this
  mean value.

### 2.7 Additional Features
- `num_studios` – raw count of production studios.
- `is_finished` – 1 if the series is completely aired.
- `duration_bucket` – Short / Standard / Long / Unknown.
- `decade` – derived from `year_raw` before imputation.

### 2.8 Column Naming Convention
- All column names are lowercase, spaces and hyphens replaced by
  underscores, special characters removed.
- Examples: `type_tv_special`, `rating_pg_13_teens_13_or_older`,
  `studio_brains_base`.

---

## 3. Policies for Downstream Consumers

- **Title Display**: Use `english_title` when non‑null; otherwise fall
  back to `title` (romanised Japanese / Chinese).
- **Synopsis Display**: When `synopsis_stub` is True, show a placeholder
  such as “Full synopsis not available” rather than the raw stub text.
- **Rating**: If `rating` is null, treat the content as “Unrated – assume
  mature” for age‑gating purposes.
- **Image Fallback**: If `image_url` points to `/static/no_image.png`,
  ensure the static asset exists on your front‑end server.