import pandas as pd
# Load processed encoded data (this already has encoded 'name')
df = pd.read_csv("data/processed/cleaned_data.csv")
# Load original raw data (this has text names)
df_raw = pd.read_csv("data/raw/BIKEDETAILS.csv")
# Only need original names
df_raw = df_raw[["name"]]
# Attach encoded name column
df_raw["encoded_name"] = df["name"]
# Remove duplicates
df_mapping = df_raw.drop_duplicates()
# Save mapping
df_mapping.to_csv("data/name_encoding.csv", index=False)
print("Saved file: data/name_encoding.csv")