# 📰 Fake News Detection using Machine Learning

This project is designed to detect whether a given news article is **real** or **fake** using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 📌 Features

- Detects fake or real news articles using Logistic Regression.
- Utilizes TF-IDF for text vectorization.
- Trained on real-world labeled news dataset.
- Includes model evaluation: accuracy, confusion matrix, and classification report.

---

## 📊 Dataset

> ⚠️ **Note**: Dataset files (`Fake.csv`, `True.csv`, and `fake_or_real_news.csv`) are not included in the repository due to size.  
> You can download them from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) and place them in the project directory.

---

## 🛠️ Project Structure

fake_news_detection/
│
├── train_model.py # Train the model and save it
├── predict.py # Predict news using saved model
├── vectorizer.pkl # Saved TF-IDF Vectorizer
├── model.pkl # Trained Logistic Regression model
├── .gitignore # Ignore dataset files and other generated files
├── requirements.txt # List of dependencies
└── README.md # Project documentation


---

## 🧠 Technologies Used

- **Python 3**
- **scikit-learn** – ML algorithms and model evaluation
- **pandas** – Data manipulation
- **numpy** – Numerical operations
- **joblib** – Model serialization
- **TF-IDF Vectorizer** – Feature extraction from text

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/arunby15/Fake_News_detection.git
cd Fake_News_detection

2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Place the Dataset
Download and place the following files in the root directory:

Fake.csv

True.csv

Or optionally use a pre-merged file: fake_or_real_news.csv

4. Train the Model
bash
Copy
Edit
python train_model.py
This will create:

model.pkl – Trained Logistic Regression model

vectorizer.pkl – TF-IDF transformer

5. Make Predictions
bash
Copy
Edit
python predict.py
Then enter your custom news article text in the console.

🔍 Sample Prediction Output
bash
Copy
Edit
Enter the news text: NASA announces new mission to Jupiter
Prediction: Real News
