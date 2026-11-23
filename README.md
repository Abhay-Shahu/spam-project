# 🛡️ Spam Shield AI - SMS & Email Spam Classifier Frontend

An interactive Machine Learning web application and API powered by a pre-trained **Multinomial Naive Bayes** classifier and **TF-IDF Vectorizer (3,000 vocabulary)**.

---

## 🚀 Two Frontend Options Included

You can launch either frontend with a single command:

### Option 1: Modern Web Application (FastAPI + TailwindCSS)
Runs a sleek, single-page application with real-time analysis, risk meters, token breakdown, and REST API docs.

```bash
python server.py
```
👉 Open your browser at: **`http://localhost:8000`**  
👉 Interactive API Swagger Docs: **`http://localhost:8000/docs`**

---

### Option 2: Streamlit Dashboard
Runs a multi-tab Streamlit dashboard with batch classification (CSV upload), live probability charts, and model insights.

```bash
streamlit run streamlit_app.py
```
👉 Automatically opens in your browser at: **`http://localhost:8501`**

---

## 📁 Project Structure

```
assignment/
├── model.pkl            # Trained Multinomial Naive Bayes classifier
├── vectorizer.pkl       # Trained TF-IDF Vectorizer (3,000 features)
├── preprocess.py        # Text cleaner (lowercase, NLTK tokenization, stopwords, Porter Stemmer)
├── server.py            # FastAPI backend + embedded modern responsive UI
├── streamlit_app.py     # Streamlit web dashboard
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🧠 Preprocessing & Inference Architecture

The vectorizer expects texts stemmed and filtered identically to its training pipeline:
1. **Lowercasing**: Normalizes casing.
2. **Tokenization**: NLTK `word_tokenize`.
3. **Alphanumeric Filtering**: Filters out special characters.
4. **Stopwords & Punctuation Removal**: Drops common English stopwords and punctuation.
5. **Stemming**: Applies `PorterStemmer()` (e.g., *congratulations* → *congratul*, *urgently* → *urgent*).
6. **TF-IDF Vectorization & MultinomialNB Inference**: Predicts class `0` (Ham / Legitimate) or `1` (Spam) with class probabilities.

---

## 🔌 REST API Endpoints

### 1. Single Prediction
- **Endpoint**: `POST /api/predict`
- **Request Body**:
  ```json
  {
    "text": "Congratulations! You have won a £1,000 Walmart Gift Card. Call 08000930705 now!"
  }
  ```
- **Response**:
  ```json
  {
    "prediction": 1,
    "verdict": "SPAM",
    "spam_probability": 98.42,
    "ham_probability": 1.58,
    "transformed_text": "congratul walmart gift card call claim",
    "matched_features": ["congratul", "gift", "card", "call", "claim"]
  }
  ```

### 2. Batch Prediction
- **Endpoint**: `POST /api/predict-batch`
- **Request Body**:
  ```json
  {
    "texts": [
      "Hey, are you free for lunch today?",
      "URGENT! Call 08452810073 to claim prize"
    ]
  }
  ```
