import pandas as pd

# Load both datasets
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Add label column
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Keep only relevant columns
true_df = true_df[['title']].rename(columns={'title': 'text'})
fake_df = fake_df[['title']].rename(columns={'title': 'text'})

# Add labels again (after renaming)
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Merge and shuffle
combined_df = pd.concat([true_df, fake_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged dataset
combined_df.to_csv('fake_or_real_news.csv', index=False)
print("✅ Dataset merged and saved as fake_or_real_news.csv")
