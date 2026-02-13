import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from langdetect import detect
from deep_translator import GoogleTranslator
import safetensors.torch

from navbar import render_navbar

st.set_page_config(
    page_title="Translation & Detection",
    layout="wide",
    page_icon="🔁"
)

render_navbar(active="detect")

# ==============================
# UI STYLING
# ==============================
st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Segoe UI", sans-serif; }
.main-title {
    font-size: 32px; text-align: center; font-weight: bold;
    color: #1f77b4; padding-top: 15px;
}
.sub-title {
    text-align: center; font-size: 16px; color: #5f6b7a;
    margin-bottom: 20px;
}
.card {
    background: rgba(255, 255, 255, 0.55);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 4px 25px rgba(0,0,0,0.08);
    backdrop-filter: blur(12px);
    margin-bottom: 25px;
}
.stButton>button {
    background-image: linear-gradient(120deg, #1f77b4, #175f8f);
    color: white;
    padding: 10px 18px;
    border-radius: 12px;
    border: none;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# TRANSLATION LANGUAGE OPTIONS
# ==============================
translation_languages = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Marathi": "mr",
    "Spanish": "es",
    "Arabic": "ar",
}

language_map = {
    "en": "English", "hi": "Hindi", "kn": "Kannada",
    "mr": "Marathi", "ml": "Malayalam", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu", "ar": "Arabic",
    "es": "Spanish", "fr": "French", "de": "German"
}

def detect_lang(text: str) -> str:
    try:
        code = detect(text)
        return language_map.get(code, "Unknown")
    except Exception:
        return "Unknown"

def translate_text(text: str, target_code: str) -> str:
    try:
        return GoogleTranslator(source='auto', target=target_code).translate(text)
    except Exception:
        return text

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_xlmr_multilingual():
    model_path = "models/transformer"

    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )
    weights = safetensors.torch.load_file(
        f"{model_path}/model.safetensors",
        device="cpu"
    )
    model.load_state_dict(weights)

    tokenizer = XLMRobertaTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model.to("cpu")
    model.eval()
    return model, tokenizer

KANNADA_DIR = "models/kannada"

@st.cache_resource
def load_kannada_svm():
    vec = joblib.load(f"{KANNADA_DIR}/kannada_svm_vectorizer.pkl")
    clf = joblib.load(f"{KANNADA_DIR}/kannada_svm.pkl")
    return vec, clf

@st.cache_resource
def load_kannada_nb():
    vec = joblib.load(f"{KANNADA_DIR}/kannada_nb_vectorizer.pkl")
    clf = joblib.load(f"{KANNADA_DIR}/kannada_nb.pkl")
    return vec, clf

class SimpleDenseNN(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

@st.cache_resource
def load_kannada_lstm():
    vec = joblib.load(f"{KANNADA_DIR}/kannada_lstm_tfidf.pkl")
    model = SimpleDenseNN(len(vec.get_feature_names_out()))
    state = torch.load(f"{KANNADA_DIR}/kannada_lstm.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return vec, model

@st.cache_resource
def load_kannada_gru():
    vec = joblib.load(f"{KANNADA_DIR}/kannada_gru_tfidf.pkl")
    model = SimpleDenseNN(len(vec.get_feature_names_out()))
    state = torch.load(f"{KANNADA_DIR}/kannada_gru.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return vec, model

@st.cache_resource
def load_kannada_xlmr():
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=2
    )
    state = torch.load(
        f"{KANNADA_DIR}/kannada_xlmr_high_accuracy.pt",
        map_location="cpu"
    )
    model.load_state_dict(state)
    model.to("cpu")
    model.eval()
    return model, tokenizer

# ==============================
# PREDICTION HELPERS
# ==============================
def predict_xlmr(text: str, model, tokenizer):
    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1)[0].cpu().numpy()
    return probs

def predict_tfidf_linear(text: str, vec, clf):
    X = vec.transform([text])
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[0]
    else:
        score = clf.decision_function(X)[0]
        p1 = 1 / (1 + np.exp(-score))
        return np.array([1 - p1, p1])

def predict_tfidf_nn(text: str, vec, model_nn):
    X = vec.transform([text])
    row = torch.tensor(X.toarray(), dtype=torch.float32)
    with torch.no_grad():
        out = model_nn(row)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return probs

# ==============================
# EXPLANATION LAYER
# ==============================
def generate_explanation(text: str, label: str) -> str:
    t = text.lower()

    hate_targets = [
        "muslim", "hindu", "christian", "dalit", "brahmin", "caste",
        "religion", "community", "minority", "immigrant",
        "woman", "girl", "gender", "lgbt", "gay", "lesbian",
        "black", "white", "race"
    ]
    targeted = any(w in t for w in hate_targets)

    if label == "Non-Hate":
        if not targeted:
            return (
                "• The sentence may be rude or negative.\n"
                "• But it does **not target any community, religion, caste, gender or race**.\n"
                "• No explicit abusive slurs are present.\n"
                "➡ Therefore, it is classified as **Non-Hate Speech** (just negative sentiment)."
            )
        else:
            return (
                "• The text mentions a group/community word.\n"
                "• However, there is no direct insult or threat.\n"
                "➡ Model still considers this as **Non-Hate Speech**."
            )
    else:  # Hate
        if targeted:
            return (
                "• The text targets a **protected group** (religion/caste/community/gender/etc.).\n"
                "• Hate speech involves attacking or insulting such identities.\n"
                "➡ Therefore, it is classified as **Hate Speech**."
            )
        else:
            return (
                "• The text shows strong abusive or insulting tone.\n"
                "• Even without explicit group references, the language is highly aggressive.\n"
                "➡ Therefore, it is marked as **Hate Speech**."
            )

# ==============================
# PAGE LAYOUT
# ==============================
st.markdown("<h1 class='main-title'>🔁 Translation & Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Multilingual Hate Speech Detection with Kannada Mode Toggle & Explanations</div>",
    unsafe_allow_html=True
)

# Input text
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### ✏️ Enter text to analyze")
text_input = st.text_area(
    "",
    height=150,
    placeholder="Type Kannada, Hindi, Hinglish, English, Arabic, Spanish etc..."
)
st.markdown("</div>", unsafe_allow_html=True)

# Translation language selection
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🌐 Select translation output language")
target_lang = st.selectbox("Choose language:", list(translation_languages.keys()))
target_code = translation_languages[target_lang]
st.markdown("</div>", unsafe_allow_html=True)

# Results section
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Detect + Translate + Predict"):
        if not text_input.strip():
            st.warning("Please enter text.")
        else:
            detected_lang = detect_lang(text_input)

            # =======================
            #  FINAL KANNADA FLOW WITH TOGGLE
            # =======================
            if detected_lang == "Kannada":
                st.info("**Detected Language:** Kannada")

                # Toggle for Kannada mode
                mode = st.radio(
                    "Choose Kannada Processing Mode:",
                    [
                        "Use Kannada Dedicated Models",
                        "Use Translation + Multilingual XLM-R",
                    ]
                )

                # ===============================
                #  OPTION 1 → USE DEDICATED KANNADA MODELS
                # ===============================
                if mode == "Use Kannada Dedicated Models":
                    st.success("Kannada mode: Using dedicated SVM/NB/LSTM/GRU/XLM-R models.")

                    # Kannada model selection
                    model_name = st.selectbox(
                        "Select Kannada Model",
                        [
                            "XLM-R Kannada (High Accuracy)",
                            "SVM",
                            "Naive Bayes",
                            "LSTM Dense",
                            "GRU Dense"
                        ]
                    )

                    text_to_use = text_input

                    if model_name == "SVM":
                        vec, clf = load_kannada_svm()
                        probs = predict_tfidf_linear(text_to_use, vec, clf)
                        model_tag = "Kannada SVM"

                    elif model_name == "Naive Bayes":
                        vec, clf = load_kannada_nb()
                        probs = predict_tfidf_linear(text_to_use, vec, clf)
                        model_tag = "Kannada Naive Bayes"

                    elif model_name == "LSTM Dense":
                        vec, model_nn = load_kannada_lstm()
                        probs = predict_tfidf_nn(text_to_use, vec, model_nn)
                        model_tag = "Kannada LSTM Dense"

                    elif model_name == "GRU Dense":
                        vec, model_nn = load_kannada_gru()
                        probs = predict_tfidf_nn(text_to_use, vec, model_nn)
                        model_tag = "Kannada GRU Dense"

                    else:
                        model_ka, tok_ka = load_kannada_xlmr()
                        probs = predict_xlmr(text_to_use, model_ka, tok_ka)
                        model_tag = "Kannada XLM-R (High Accuracy)"

                    pred = int(np.argmax(probs))
                    conf = float(probs[pred] * 100)
                    label = "Hate" if pred == 1 else "Non-Hate"

                    st.markdown("### 🧠 Prediction Result")
                    st.caption(f"Model used: {model_tag}")

                    if pred == 1:
                        st.error(f"⚠️ Hate Speech — **{conf:.2f}%**")
                    else:
                        st.success(f"✅ Non-Hate Speech — **{conf:.2f}%**")

                    explanation = generate_explanation(text_to_use, label)
                    st.markdown("### 📝 Explanation")
                    st.info(explanation)

                    st.stop()   # prevent multilingual flow

                # ===============================
                #  OPTION 2 → TRANSLATION + MULTILINGUAL XLM-R
                # ===============================
                else:
                    st.success("Kannada mode: Using translation → Multilingual XLM-R")

                    # Translate Kannada → selected output language
                    translated = translate_text(text_input, target_code)
                    st.success(f"Translated Text ({target_lang}): {translated}")

                    # For prediction we use English translation internally
                    translated_for_model = translate_text(text_input, "en")

                    model, tokenizer = load_xlmr_multilingual()
                    probs = predict_xlmr(translated_for_model, model, tokenizer)

                    pred = int(np.argmax(probs))
                    conf = float(probs[pred] * 100)
                    label = "Hate" if pred == 1 else "Non-Hate"

                    st.markdown("### 🧠 Prediction Result")
                    st.caption("Model used: Multilingual XLM-R (Translated from Kannada)")

                    if pred == 1:
                        st.error(f"⚠️ Hate Speech — **{conf:.2f}%**")
                    else:
                        st.success(f"✅ Non-Hate Speech — **{conf:.2f}%**")

                    explanation = generate_explanation(translated_for_model, label)
                    st.markdown("### 📝 Explanation")
                    st.info(explanation)

                    st.stop()

            # =======================
            #  NON-KANNADA FLOW
            # =======================
            st.info(f"**Detected Language:** {detected_lang}")

            # 1) Translate to user-selected display language
            translated_for_user = translate_text(text_input, target_code)
            st.success(f"Translated Text ({target_lang}): {translated_for_user}")

            # 2) Translate to English for model inference
            translated_for_model = translate_text(text_input, "en")

            # 3) Multilingual model prediction
            model, tokenizer = load_xlmr_multilingual()
            probs = predict_xlmr(translated_for_model, model, tokenizer)

            pred = int(np.argmax(probs))
            conf = float(probs[pred] * 100)
            label = "Hate" if pred == 1 else "Non-Hate"

            st.markdown("### 🧠 Prediction Result")
            st.caption("Model used: Multilingual XLM-R")

            if pred == 1:
                st.error(f"⚠️ Hate Speech — **{conf:.2f}%**")
            else:
                st.success(f"✅ Non-Hate Speech — **{conf:.2f}%**")

            explanation = generate_explanation(translated_for_model, label)
            st.markdown("### 📝 Explanation")
            st.info(explanation)

with col2:
    st.markdown("### ℹ️ Notes")
    st.write(
        """
- The app **detects language**, then:
  - If **Kannada**, you can choose:
    - Dedicated Kannada models (SVM/NB/LSTM/GRU/XLM-R), OR  
    - Translation + Multilingual XLM-R.
  - If **Non-Kannada**, it:
    - Translates text to your selected language for display.
    - Translates text to English internally for XLM-R prediction.
- The explanation section describes **why** it is Hate or Non-Hate:
  - Checks for group-related words (religion, caste, gender, etc.).
  - Distinguishes between **hate speech** and **normal negative sentiment**.
        """
    )

st.markdown("</div>", unsafe_allow_html=True)
