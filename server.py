import os
import pickle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List
from preprocess import transform_text

app = FastAPI(
    title="Spam Shield AI - API & Frontend",
    description="Multinomial Naive Bayes + TF-IDF Spam Detection Service",
    version="1.0.0"
)

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

vocab_set = set(vectorizer.get_feature_names_out())

class PredictRequest(BaseModel):
    text: str

class BatchPredictRequest(BaseModel):
    texts: List[str]

@app.post("/api/predict")
def predict_single(req: PredictRequest):
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Text cannot be empty"})
    
    transformed = transform_text(req.text)
    tokens = transformed.split() if transformed else []
    matched_features = [t for t in tokens if t in vocab_set]
    
    X_vec = vectorizer.transform([transformed])
    pred = int(model.predict(X_vec)[0])
    probas = model.predict_proba(X_vec)[0]
    
    return {
        "text": req.text,
        "transformed_text": transformed,
        "tokens": tokens,
        "matched_features": matched_features,
        "prediction": pred,
        "verdict": "SPAM" if pred == 1 else "HAM",
        "spam_probability": round(float(probas[1]) * 100, 2),
        "ham_probability": round(float(probas[0]) * 100, 2)
    }

@app.post("/api/predict-batch")
def predict_batch(req: BatchPredictRequest):
    results = []
    for text in req.texts:
        if not text.strip():
            continue
        transformed = transform_text(text)
        X_vec = vectorizer.transform([transformed])
        pred = int(model.predict(X_vec)[0])
        probas = model.predict_proba(X_vec)[0]
        results.append({
            "text": text,
            "verdict": "SPAM" if pred == 1 else "HAM",
            "spam_probability": round(float(probas[1]) * 100, 2),
            "ham_probability": round(float(probas[0]) * 100, 2)
        })
    return {"results": results, "total": len(results)}

@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTML_CONTENT

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Shield AI - Message Classifier</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['"Plus Jakarta Sans"', 'sans-serif'],
                    },
                    colors: {
                        brand: {
                            50: '#eef2ff',
                            100: '#e0e7ff',
                            500: '#6366f1',
                            600: '#4f46e5',
                            700: '#4338ca',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        .glass {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }
        .gradient-text {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen flex flex-col antialiased">
    <!-- Navbar -->
    <header class="border-b border-slate-200 bg-white sticky top-0 z-50 shadow-sm">
        <div class="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center text-white shadow-md">
                    <i data-lucide="shield-check" class="w-6 h-6"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight text-slate-800">SpamShield <span class="text-indigo-600">AI</span></h1>
                    <p class="text-xs text-slate-500 font-medium">Multinomial Naive Bayes & TF-IDF Classifier</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 mr-2 animate-pulse"></span> Model Active (3,000 Vocab)
                </span>
                <a href="/docs" target="_blank" class="text-xs font-semibold text-slate-600 hover:text-indigo-600 transition flex items-center space-x-1 border border-slate-200 px-3 py-1.5 rounded-lg bg-slate-50">
                    <span>API Docs</span>
                    <i data-lucide="external-link" class="w-3.5 h-3.5"></i>
                </a>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-6xl mx-auto px-4 py-8 flex-1 w-full">
        <!-- Hero Section -->
        <div class="text-center max-w-2xl mx-auto mb-8">
            <h2 class="text-3xl sm:text-4xl font-extrabold tracking-tight mb-3">
                Inspect Messages with <span class="gradient-text">Precision ML</span>
            </h2>
            <p class="text-slate-600 text-sm sm:text-base">
                Detect phishing scams, fraudulent prizes, and unsolicited SMS spam in milliseconds with our trained NLP text classification pipeline.
            </p>
        </div>

        <!-- Quick Examples -->
        <div class="flex flex-wrap items-center justify-center gap-2 mb-6">
            <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider mr-2">Try Samples:</span>
            <button onclick="setSample(1)" class="text-xs font-medium px-3 py-1.5 rounded-full bg-red-50 text-red-700 border border-red-200 hover:bg-red-100 transition flex items-center space-x-1.5">
                <i data-lucide="alert-triangle" class="w-3.5 h-3.5 text-red-500"></i>
                <span>Lottery Winner Scam</span>
            </button>
            <button onclick="setSample(2)" class="text-xs font-medium px-3 py-1.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100 transition flex items-center space-x-1.5">
                <i data-lucide="clock" class="w-3.5 h-3.5 text-amber-500"></i>
                <span>Urgent Service Expiry</span>
            </button>
            <button onclick="setSample(3)" class="text-xs font-medium px-3 py-1.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 transition flex items-center space-x-1.5">
                <i data-lucide="message-square" class="w-3.5 h-3.5 text-emerald-500"></i>
                <span>Casual Meeting Chat</span>
            </button>
            <button onclick="setSample(4)" class="text-xs font-medium px-3 py-1.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100 transition flex items-center space-x-1.5">
                <i data-lucide="check-circle" class="w-3.5 h-3.5 text-blue-500"></i>
                <span>Normal Friendly Message</span>
            </button>
        </div>

        <!-- Input & Results Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <!-- Left Col: Input Box -->
            <div class="lg:col-span-7 bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
                <div class="flex items-center justify-between mb-3">
                    <label for="messageInput" class="text-sm font-bold text-slate-700 flex items-center space-x-2">
                        <i data-lucide="message-circle" class="w-4 h-4 text-indigo-600"></i>
                        <span>Input Message / SMS / Email Excerpt</span>
                    </label>
                    <span id="charCount" class="text-xs text-slate-400 font-mono">0 chars</span>
                </div>
                
                <textarea id="messageInput" rows="6" 
                    placeholder="Enter or paste any suspicious or normal message here..."
                    class="w-full rounded-xl border border-slate-300 p-4 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition resize-none"></textarea>
                
                <div class="mt-4 flex items-center justify-between">
                    <button onclick="clearText()" class="text-xs font-semibold text-slate-500 hover:text-slate-800 transition px-3 py-2">
                        Clear text
                    </button>
                    <button id="analyzeBtn" onclick="analyzeMessage()" class="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-semibold text-sm rounded-xl shadow-md hover:shadow-lg transition flex items-center space-x-2">
                        <i data-lucide="sparkles" class="w-4 h-4"></i>
                        <span>Analyze Message</span>
                    </button>
                </div>

                <!-- Internal Processing Details -->
                <div id="tokenSection" class="mt-6 pt-5 border-t border-slate-100 hidden">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Preprocessing & Stemmed Tokens</h4>
                    <p id="stemmedText" class="text-xs font-mono bg-slate-50 p-2.5 rounded-lg border border-slate-200 text-slate-700 mb-3"></p>
                    <div class="text-xs text-slate-500 mb-2">Tokens matched in model vocabulary:</div>
                    <div id="tokenChips" class="flex flex-wrap gap-1.5"></div>
                </div>
            </div>

            <!-- Right Col: Verdict Card -->
            <div class="lg:col-span-5">
                <div id="verdictCard" class="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 text-center">
                    <div id="emptyState" class="py-12">
                        <div class="w-16 h-16 rounded-full bg-slate-100 mx-auto flex items-center justify-center text-slate-400 mb-4">
                            <i data-lucide="scan" class="w-8 h-8"></i>
                        </div>
                        <h3 class="text-base font-semibold text-slate-700">Awaiting Analysis</h3>
                        <p class="text-xs text-slate-400 mt-1 max-w-xs mx-auto">Click one of the quick samples above or type in a message and press Analyze.</p>
                    </div>

                    <!-- Result Display -->
                    <div id="resultState" class="hidden text-left">
                        <div id="verdictBadge" class="p-4 rounded-xl flex items-center space-x-3 mb-6">
                            <div id="verdictIcon" class="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"></div>
                            <div>
                                <h3 id="verdictTitle" class="text-lg font-extrabold leading-tight"></h3>
                                <p id="verdictSub" class="text-xs font-medium"></p>
                            </div>
                        </div>

                        <!-- Confidence Bars -->
                        <div class="space-y-4">
                            <div>
                                <div class="flex justify-between text-xs font-bold mb-1">
                                    <span class="text-red-700 flex items-center space-x-1">
                                        <i data-lucide="alert-octagon" class="w-3.5 h-3.5"></i>
                                        <span>Spam Risk Probability</span>
                                    </span>
                                    <span id="spamProbText" class="font-mono text-red-600">0%</span>
                                </div>
                                <div class="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                                    <div id="spamBar" class="bg-gradient-to-r from-orange-500 to-red-600 h-3 rounded-full transition-all duration-700 ease-out" style="width: 0%"></div>
                                </div>
                            </div>

                            <div>
                                <div class="flex justify-between text-xs font-bold mb-1">
                                    <span class="text-emerald-700 flex items-center space-x-1">
                                        <i data-lucide="check-circle-2" class="w-3.5 h-3.5"></i>
                                        <span>Legitimate (Ham) Probability</span>
                                    </span>
                                    <span id="hamProbText" class="font-mono text-emerald-600">0%</span>
                                </div>
                                <div class="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                                    <div id="hamBar" class="bg-gradient-to-r from-teal-500 to-emerald-600 h-3 rounded-full transition-all duration-700 ease-out" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Technical Specs -->
                        <div class="mt-6 pt-4 border-t border-slate-100 grid grid-cols-2 gap-2 text-center text-xs">
                            <div class="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                                <span class="text-slate-400 block text-[10px] uppercase font-semibold">Classifier</span>
                                <span class="font-bold text-slate-700">MultinomialNB</span>
                            </div>
                            <div class="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                                <span class="text-slate-400 block text-[10px] uppercase font-semibold">Vectorizer</span>
                                <span class="font-bold text-slate-700">TF-IDF (3k vocab)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-200 bg-white py-4 text-center text-xs text-slate-500">
        Spam Classifier Frontend &bull; Powered by Scikit-Learn, FastAPI & NLTK
    </footer>

    <script>
        lucide.createIcons();

        const samples = {
            1: "CONGRATULATIONS! You have won a £1,000 Walmart Gift Card. Call 08000930705 now or click to claim immediately! T&Cs apply.",
            2: "URGENT! Your mobile service is expiring today. Reply STOP to cancel or call 08452810073 to renew now.",
            3: "Hey, can we push our sync call to 4 PM? Let me know if that works for you.",
            4: "I am on the way home, pick up some groceries please."
        };

        const input = document.getElementById('messageInput');
        const charCount = document.getElementById('charCount');

        input.addEventListener('input', () => {
            charCount.textContent = `${input.value.length} chars`;
        });

        function setSample(id) {
            input.value = samples[id];
            charCount.textContent = `${input.value.length} chars`;
            analyzeMessage();
        }

        function clearText() {
            input.value = "";
            charCount.textContent = `0 chars`;
            document.getElementById('emptyState').classList.remove('hidden');
            document.getElementById('resultState').classList.add('hidden');
            document.getElementById('tokenSection').classList.add('hidden');
        }

        async function analyzeMessage() {
            const text = input.value.trim();
            if (!text) {
                alert("Please enter some text to analyze.");
                return;
            }

            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.innerHTML = `<span class="animate-spin mr-2">&#9696;</span> Analyzing...`;

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();

                if (res.ok) {
                    renderResult(data);
                } else {
                    alert(data.error || "Prediction failed.");
                }
            } catch (err) {
                alert("Network error: " + err);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<i data-lucide="sparkles" class="w-4 h-4"></i><span>Analyze Message</span>`;
                lucide.createIcons();
            }
        }

        function renderResult(data) {
            document.getElementById('emptyState').classList.add('hidden');
            document.getElementById('resultState').classList.remove('hidden');

            const isSpam = data.prediction === 1;
            const badge = document.getElementById('verdictBadge');
            const icon = document.getElementById('verdictIcon');
            const title = document.getElementById('verdictTitle');
            const sub = document.getElementById('verdictSub');

            if (isSpam) {
                badge.className = "p-4 rounded-xl flex items-center space-x-3 mb-6 bg-red-50 border border-red-200 text-red-900";
                icon.className = "w-10 h-10 rounded-lg flex items-center justify-center shrink-0 bg-red-600 text-white";
                icon.innerHTML = `<i data-lucide="alert-triangle" class="w-6 h-6"></i>`;
                title.textContent = "SPAM DETECTED";
                sub.textContent = "Unsolicited, high-risk, or fraudulent pattern identified";
            } else {
                badge.className = "p-4 rounded-xl flex items-center space-x-3 mb-6 bg-emerald-50 border border-emerald-200 text-emerald-900";
                icon.className = "w-10 h-10 rounded-lg flex items-center justify-center shrink-0 bg-emerald-600 text-white";
                icon.innerHTML = `<i data-lucide="shield-check" class="w-6 h-6"></i>`;
                title.textContent = "LEGITIMATE (HAM)";
                sub.textContent = "Safe, genuine personal or professional communication";
            }

            document.getElementById('spamProbText').textContent = `${data.spam_probability}%`;
            document.getElementById('hamProbText').textContent = `${data.ham_probability}%`;
            document.getElementById('spamBar').style.width = `${data.spam_probability}%`;
            document.getElementById('hamBar').style.width = `${data.ham_probability}%`;

            // Tokens
            const tokenSection = document.getElementById('tokenSection');
            tokenSection.classList.remove('hidden');
            document.getElementById('stemmedText').textContent = data.transformed_text || "(empty)";

            const chipsContainer = document.getElementById('tokenChips');
            chipsContainer.innerHTML = "";
            if (data.matched_features && data.matched_features.length > 0) {
                data.matched_features.forEach(feat => {
                    const span = document.createElement('span');
                    span.className = "px-2 py-1 rounded bg-indigo-50 border border-indigo-200 text-indigo-700 text-xs font-mono font-medium";
                    span.textContent = feat;
                    chipsContainer.appendChild(span);
                });
            } else {
                chipsContainer.innerHTML = `<span class="text-xs text-slate-400 italic">No features in 3,000 TF-IDF dictionary matched.</span>`;
            }

            lucide.createIcons();
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
