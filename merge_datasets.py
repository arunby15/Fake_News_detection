import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Data
df = pd.read_csv('fake_or_real_news.csv')  # change to your file name

# Step 2: Check and clean data
print(df.head())
print(df['label'].value_counts())  # Check balance

# Optional: if labels are strings ('FAKE', 'REAL'), convert them to binary
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Step 3: Feature and Label split
X = df['text']  # Or 'title' if you want to use titles
y = df['label']

# Step 4: Vectorize Text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Step 5: Split data (stratify ensures both labels exist in train/test)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, stratify=y, random_state=42)

# Step 6: Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
