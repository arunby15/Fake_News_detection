from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import os

app = Flask(__name__)
app.secret_key = 'admin123'  # For session encryption

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Dummy login credentials
USERNAME = 'admin'
PASSWORD = 'admin123'

@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] == USERNAME and request.form['password'] == PASSWORD:
            session['user'] = USERNAME
            return redirect('/predict')
        else:
            error = 'Invalid credentials. Try again.'
    return render_template('login.html', error=error)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')

    prediction = None
    if request.method == 'POST':
        text = request.form['news']
        vectorized = vectorizer.transform([text])
        result = model.predict(vectorized)
        print(f"🔍 Prediction result: {result}")  # Debug output

        prediction = 'Real News' if result[0] == 1 else 'Fake News'
    return render_template('index.html', prediction=prediction)

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return redirect('/login')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
