import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Page config
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠", layout="wide")

# Custom CSS styles for side images and layout
st.markdown("""
    <style>
        .side-left {
            position: fixed;
            top: 100px;
            left: 10px;
            width: 100px;
            z-index: 1;
        }
        .side-right {
            position: fixed;
            top: 100px;
            right: 10px;
            width: 100px;
            z-index: 1;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #333333;
        }
        .subtitle {
            text-align: center;
            color: #666666;
            font-size: 18px;
        }
    </style>

    <div class="side-left">
        <img src="https://img.icons8.com/color/96/news.png" width="80">
    </div>
    <div class="side-right">
        <img src="https://img.icons8.com/color/96/artificial-intelligence.png" width="80">
    </div>
""", unsafe_allow_html=True)

# NLP Preprocessing Functions
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def process_text(text):
    try:
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except:
        return text

# Title and Description
st.markdown('<p class="title">🧠 Fake News Detection App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Paste a news article below to find out whether it\'s Real or Fake</p>', unsafe_allow_html=True)
st.markdown("---")

# User Input
st.markdown("### 📝 Enter News Article:")
user_input = st.text_area(
    label="Enter news content below",
    label_visibility="collapsed",  # hides label visually but keeps it for accessibility
    height=200,
    placeholder="Paste your news text here..."
)
# Sidebar for additional information
st.sidebar.markdown("### 📚 About the Model")
st.sidebar.markdown("""This model uses a Logistic Regression classifier trained on a dataset of news articles to predict whether a given article is real or fake. It preprocesses the text by cleaning, tokenizing, removing stop words, and lemmatizing before making predictions.""") 
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.markdown("""- **Accuracy:** 98%
- **Precision:** 96%    
- **Recall:** 98%
- **F1 Score:** 95%""")
# Sidebar for additional resources
st.sidebar.markdown("### 🔗 Resources")
st.sidebar.markdown("### 📚 References")
st.sidebar.markdown("""- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Streamlit Documentation](https://docs.streamlit.io/)""")

# Sidebar for additional information
st.sidebar.markdown("### 📧 Contact")
st.sidebar.markdown("For any questions or feedback, please reach out to us at [](mailto:abc@gmail.com)")
st.sidebar.markdown("### 🛠️ Tools Used")
st.sidebar.markdown("""- **Streamlit**: For building the web app interface.
- **NLTK**: For natural language processing tasks.
- **Scikit-learn**: For machine learning model training and prediction.
- **Joblib**: For model serialization and deserialization.""")


# Prediction button
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Preprocessing
        cleaned = clean_text(user_input)
        processed = process_text(cleaned)

        # Vectorization & Prediction
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0][prediction]

        # Display results
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 🔎 Result:")
            if prediction == 1:
                st.success("🟢 The news is predicted to be **REAL**.")
            else:
                st.error("🔴 The news is predicted to be **FAKE**.")

        with col2:
            st.markdown("### 📊 Confidence Level:")
            st.progress(min(max(confidence, 0.01), 0.99))
            st.info(f"**Confidence Score:** `{confidence:.2%}`")

        # Expandable preprocessed view
        with st.expander("📄 View Preprocessed Text"):
            st.code(processed, language="text")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: grey;">Made with ❤️ using Streamlit, NLTK & Scikit-learn</p>', unsafe_allow_html=True)
