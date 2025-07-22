import pandas as pd

# Load the fake and real datasets
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Add a label column: 'FAKE' for fake news, 'REAL' for real news
fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

# Combine the datasets
combined_df = pd.concat([fake_df, real_df], ignore_index=True)

# Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset to a CSV file
combined_df.to_csv("fake_or_real_news.csv", index=False)

print("✅ Merged dataset saved as fake_or_real_news.csv")
