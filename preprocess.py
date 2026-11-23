import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK resources are available
for pkg in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

ps = PorterStemmer()
stop_words_set = set(stopwords.words('english'))

def transform_text(text: str) -> str:
    """
    Preprocess raw text to match the exact pipeline used to train vectorizer.pkl:
    1. Lowercase conversion
    2. Tokenization via NLTK
    3. Alphanumeric filtering
    4. Stopwords & punctuation removal
    5. Porter stemming
    """
    if not text:
        return ""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    # Keep only alphanumeric tokens
    filtered = [token for token in tokens if token.isalnum()]
    
    # Remove stopwords and punctuation
    filtered = [token for token in filtered if token not in stop_words_set and token not in string.punctuation]
    
    # Apply Porter Stemmer
    stemmed = [ps.stem(token) for token in filtered]
    
    return " ".join(stemmed)
