import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from preprocess import transform_text

# Page config
st.set_page_config(
    page_title="Spam Shield AI - Message Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
    }
    
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .spam-badge {
        background-color: #fee2e2;
        color: #ef4444;
        border: 1px solid #fca5a5;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    
    .ham-badge {
        background-color: #dcfce7;
        color: #16a34a;
        border: 1px solid #86efac;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    
    .token-chip {
        display: inline-block;
        background: #ede9fe;
        color: #6d28d9;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vec_path = os.path.join(base_dir, "vectorizer.pkl")
    model_path = os.path.join(base_dir, "model.pkl")
    
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    return vectorizer, model

try:
    vectorizer, model = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Error loading model files: {e}")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/security-shield.png", width=70)
    st.title("Model Intelligence")
    st.caption("Powered by Scikit-Learn MultinomialNB & TF-IDF")
    
    st.markdown("---")
    if artifacts_loaded:
        vocab_size = len(vectorizer.get_feature_names_out())
        st.metric(label="Model Architecture", value="Multinomial Naive Bayes")
        st.metric(label="TF-IDF Vocabulary Size", value=f"{vocab_size:,} tokens")
        st.metric(label="Input Preprocessor", value="Porter Stemmer + NLTK")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Try Examples")
    preset_spam = "CONGRATULATIONS! You have won a £1,000 Walmart Gift Card. Call 08000930705 now or click to claim immediately! T&Cs apply."
    preset_ham = "Hey, can we push our sync call to 4 PM? Let me know if that works for you."
    preset_urgency = "URGENT! Your mobile service is expiring today. Reply STOP to cancel or call 08452810073 to renew now."
    preset_casual = "I am on the way home, pick up some groceries please."
    
    if st.button("🚨 Lottery Spam Example", use_container_width=True):
        st.session_state["input_text"] = preset_spam
    if st.button("⚠️ Urgent Account Spam", use_container_width=True):
        st.session_state["input_text"] = preset_urgency
    if st.button("💬 Normal Chat Message", use_container_width=True):
        st.session_state["input_text"] = preset_ham
    if st.button("🛒 Casual Errand Note", use_container_width=True):
        st.session_state["input_text"] = preset_casual

# Main Header
st.markdown('<h1 class="main-header">🛡️ Spam Shield AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time SMS & Email Spam Detection with Naive Bayes and TF-IDF Feature Extraction</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Message Detector", "📑 Batch Classification", "📊 Vocabulary Insights"])

with tab1:
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    user_text = st.text_area(
        "Enter or paste the message text to inspect:",
        value=st.session_state.get("input_text", ""),
        height=140,
        placeholder="Type or paste any SMS, email excerpt, or notification message here..."
    )

    col_btn, col_clear = st.columns([1, 6])
    with col_btn:
        predict_clicked = st.button("🚀 Analyze Message", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear Input", use_container_width=False):
            st.session_state["input_text"] = ""
            st.rerun()

    if (predict_clicked or user_text) and artifacts_loaded:
        if user_text.strip():
            # 1. Preprocess text
            transformed = transform_text(user_text)
            
            # 2. Vectorize
            X_vec = vectorizer.transform([transformed])
            
            # 3. Predict & Proba
            prediction = model.predict(X_vec)[0]
            probas = model.predict_proba(X_vec)[0]
            
            ham_prob = probas[0] * 100
            spam_prob = probas[1] * 100
            
            st.markdown("---")
            
            # Verdict Section
            col_res, col_chart = st.columns([1.2, 1])
            
            with col_res:
                if prediction == 1:
                    st.error("### 🚨 Prediction: SPAM MESSAGE DETECTED")
                    st.markdown(
                        f"""
                        <div style="background: #fef2f2; border-left: 5px solid #ef4444; padding: 15px; border-radius: 8px;">
                            <strong style="color: #991b1b; font-size: 1.15rem;">High Risk Detected!</strong><br>
                            <span style="color: #4b5563;">This message contains patterns, keywords, or phishing lures commonly associated with fraudulent or unsolicited spam.</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.success("### ✅ Prediction: LEGITIMATE MESSAGE (HAM)")
                    st.markdown(
                        f"""
                        <div style="background: #f0fdf4; border-left: 5px solid #22c55e; padding: 15px; border-radius: 8px;">
                            <strong style="color: #166534; font-size: 1.15rem;">Safe & Legitimate!</strong><br>
                            <span style="color: #4b5563;">This message appears natural and does not exhibit common spam or phishing characteristics.</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.write("")
                st.write("**Confidence Breakdown:**")
                st.write(f"• **Spam Confidence:** `{spam_prob:.2f}%`")
                st.progress(float(probas[1]))
                st.write(f"• **Legitimate (Ham) Confidence:** `{ham_prob:.2f}%`")
                st.progress(float(probas[0]))
            
            with col_chart:
                st.subheader("Probabilistic Distribution")
                chart_df = pd.DataFrame({
                    "Class": ["Legitimate (Ham)", "Spam"],
                    "Probability (%)": [ham_prob, spam_prob]
                })
                st.bar_chart(chart_df.set_index("Class"), color="#4f46e5", height=240)
            
            # Explainability Section
            st.markdown("### 🧩 Pipeline Transparency & Tokens")
            with st.expander("Show internal token transformations and matched vocabulary", expanded=True):
                tokens = transformed.split() if transformed else []
                st.markdown(f"**Cleaned & Stemmed Representation:** `{transformed or '(none)'}`")
                
                st.markdown("**Processed Token Chips:**")
                if tokens:
                    token_html = "".join([f'<span class="token-chip">{t}</span>' for t in tokens])
                    st.markdown(token_html, unsafe_allow_html=True)
                else:
                    st.info("No alphanumeric tokens remained after stopword removal.")
                
                # Active features in TF-IDF
                vocab = vectorizer.vocabulary_
                matched_features = [t for t in tokens if t in vocab]
                st.markdown(f"**Active Features Matched in TF-IDF Dictionary ({len(matched_features)}/{len(tokens)}):**")
                if matched_features:
                    st.code(", ".join(matched_features), language="text")
                else:
                    st.warning("None of the tokens matched the training vocabulary.")
        else:
            st.warning("Please type or select a message to analyze.")

with tab2:
    st.subheader("Batch Message Classification")
    st.write("Upload a CSV file or paste multiple messages to evaluate them all at once.")
    
    batch_mode = st.radio("Choose Input Mode:", ["Paste multiple lines", "Upload CSV"], horizontal=True)
    
    if batch_mode == "Paste multiple lines":
        batch_text = st.text_area(
            "Enter messages (one per line):",
            height=150,
            value="Congratulations! You have won a $1,000 Walmart Gift Card. Call now!\nHey bro, are we meeting up for dinner tonight?\nURGENT alert: your bank account is locked, click http://verify.xyz\nThanks for your email, I will review the report and reply."
        )
        if st.button("Evaluate Batch", key="batch_paste_btn"):
            lines = [line.strip() for line in batch_text.strip().split("\n") if line.strip()]
            if lines and artifacts_loaded:
                transformed_lines = [transform_text(line) for line in lines]
                X_batch = vectorizer.transform(transformed_lines)
                preds = model.predict(X_batch)
                probas = model.predict_proba(X_batch)
                
                results = []
                for original, p, prob in zip(lines, preds, probas):
                    results.append({
                        "Message": original,
                        "Verdict": "🚨 SPAM" if p == 1 else "✅ HAM",
                        "Spam Probability (%)": round(prob[1] * 100, 2),
                        "Ham Probability (%)": round(prob[0] * 100, 2)
                    })
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Summary
                spam_count = (preds == 1).sum()
                st.metric("Spam Rate", f"{spam_count}/{len(lines)} ({spam_count/len(lines)*100:.1f}%)")
    else:
        uploaded_file = st.file_uploader("Upload CSV (must contain a column named 'text' or 'message')", type=["csv"])
        if uploaded_file and artifacts_loaded:
            try:
                df = pd.read_csv(uploaded_file)
                text_col = None
                for col in ["text", "message", "sms", "content", "body"]:
                    if col in df.columns:
                        text_col = col
                        break
                if not text_col:
                    text_col = df.columns[0]
                
                st.info(f"Using column **'{text_col}'** for analysis.")
                with st.spinner("Processing batch..."):
                    texts = df[text_col].astype(str).tolist()
                    transformed_texts = [transform_text(t) for t in texts]
                    X_b = vectorizer.transform(transformed_texts)
                    preds = model.predict(X_b)
                    probas = model.predict_proba(X_b)
                    
                    df["Verdict"] = ["SPAM" if p == 1 else "HAM" for p in preds]
                    df["Spam_Confidence_%"] = np.round(probas[:, 1] * 100, 2)
                    df["Ham_Confidence_%"] = np.round(probas[:, 0] * 100, 2)
                    
                    st.dataframe(df, use_container_width=True)
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Classified CSV", csv_data, "classified_messages.csv", "text/csv")
            except Exception as ex:
                st.error(f"Error processing CSV: {ex}")

with tab3:
    st.subheader("Model & Vocabulary Insights")
    if artifacts_loaded:
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🚨 Top Indicative Spam Features")
            top_spam_idx = np.argsort(model.feature_log_prob_[1])[-20:][::-1]
            top_spam_df = pd.DataFrame({
                "Stemmed Feature": feature_names[top_spam_idx],
                "Log Probability": np.round(model.feature_log_prob_[1][top_spam_idx], 3)
            })
            st.dataframe(top_spam_df, use_container_width=True)
            
        with col2:
            st.markdown("#### 💬 Top Indicative Ham Features")
            top_ham_idx = np.argsort(model.feature_log_prob_[0])[-20:][::-1]
            top_ham_df = pd.DataFrame({
                "Stemmed Feature": feature_names[top_ham_idx],
                "Log Probability": np.round(model.feature_log_prob_[0][top_ham_idx], 3)
            })
            st.dataframe(top_ham_df, use_container_width=True)
