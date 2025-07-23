from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        news = request.form['news']

        if not news.strip():
            return render_template('index.html', prediction="⚠️ Please enter valid news text.")

        # Vectorize input
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)
        confidence = model.decision_function(news_vector)

        return render_template(
            'index.html',
            prediction=f"📰 This news is predicted as: {prediction[0]}",
            confidence=f"📊 Confidence Score: {round(confidence[0], 2)}"
        )

# Run app
if __name__ == '__main__':
    app.run(debug=True)
