import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load data
df = pd.read_csv('fake_or_real_news.csv')

# Step 2: Convert labels to binary
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Step 3: Feature and label split
X = df['text']
y = df['label']

# Step 4: Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Step 5: Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, stratify=y, random_state=42)

# Step 6: Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n🧪 Classification Report:")
print(classification_report(y_test, y_pred))
