import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load the dataset
df = pd.read_csv("fake_or_real_news.csv")  # Ensure this file exists in the same directory

# Step 2: Keep only relevant columns
# The dataset must have 'text' and 'label' columns
df = df[['text', 'label']]

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.25, random_state=42)

# Step 4: Convert text data to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train the PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully with accuracy: {accuracy * 100:.2f}%\n")
print("🧪 Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("📦 model.pkl and vectorizer.pkl saved successfully.")
