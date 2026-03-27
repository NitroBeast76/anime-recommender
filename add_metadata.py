import pandas as pd

df = pd.read_csv("anime_data.csv", encoding="utf-8")

# Add new columns with placeholders
df["synopsis"] = "No synopsis available yet."
df["genres"] = "Action, Adventure"   # placeholder
df["release_date"] = "Unknown"
df["author"] = "Unknown"
df["studio"] = "Unknown"
df["episodes"] = 0
df["rating"] = "N/A"

# Save back
df.to_csv("anime_data.csv", index=False, encoding="utf-8")
print("Added metadata columns.")