from flask import Flask, render_template, request
import pickle

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Home route (renders form)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        news = request.form['news']
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)
        return render_template('index.html', prediction=prediction[0])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
