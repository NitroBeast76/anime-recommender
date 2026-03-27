import pandas as pd

# Read the existing CSV (without image_url)
df = pd.read_csv("anime_data.csv", encoding="utf-8")

# Add a new column with the placeholder URL
placeholder_url = "https://placehold.co/600x400?text=Hello+World"
df["image_url"] = placeholder_url

# Save back to the same file
df.to_csv("anime_data.csv", index=False, encoding="utf-8")
print("Done! Added image_url column.")
