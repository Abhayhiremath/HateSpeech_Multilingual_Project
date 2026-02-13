import streamlit as st
import torch
import numpy as np
import langid
import re
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import safetensors.torch
from navbar import render_navbar

st.set_page_config(page_title="Detection", layout="wide", page_icon="🔍")
render_navbar(active="detect")

st.markdown("<h1 class='main-title'>🔍 Translation & Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>Detect language, translate text, identify harmful & hate speech, and view detailed explanation.</p>",
    unsafe_allow_html=True
)

# ==========================================================
# 🔧 TEXT NORMALIZATION (ONLY FOR RULE CHECKS)
# ==========================================================
def clean_special_chars(text: str):
    if not text:
        return text

    replacements = {
        "@": "a", "0": "o", "1": "i", "$": "s", "3": "e",
        "!": "", "*": "", "#": "", "%": "", "&": ""
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================================
# 🔧 TRUE INPUT LANGUAGE DETECTION (FIXED)
# ==========================================================
def detect_language_original(text: str):
    if not text.strip():
        return "Unknown"

    # ---------- Script-based detection ----------
    if re.search(r'[\u0900-\u097F]', text):
        return "Hindi"

    if re.search(r'[\u0C80-\u0CFF]', text):
        return "Kannada"

    if re.search(r'[\u0600-\u06FF]', text):
        return "Arabic"

    # ---------- Hinglish (Roman Hindi) ----------
    hinglish_markers = [
        "hai", "hain", "kar", "karte", "rehna",
        "galat", "hata", "dena", "log", "yahan"
    ]
    lower = text.lower()
    if any(w in lower.split() for w in hinglish_markers):
        return "Hinglish (Romanized Hindi)"

    # ---------- langid fallback ----------
    code, _ = langid.classify(text)

    mapping = {
        "en": "English",
        "hi": "Hindi",
        "kn": "Kannada",
        "mr": "Marathi",
        "es": "Spanish",
        "ar": "Arabic"
    }

    return mapping.get(code, "Unknown")


# =====================================================
# UI LAYOUT
# =====================================================
left_col, right_col = st.columns([1.2, 1])

with left_col:
    text = st.text_area("Enter text to analyze:", height=220)

    translation_options = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
        "Spanish": "es",
        "Arabic": "ar",
        "Marathi": "mr"
    }
    target_lang = st.selectbox("Translate output to:", list(translation_options.keys()))
    target_code = translation_options[target_lang]

    analyze_btn = st.button("Analyze", use_container_width=True)


# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_multilingual_model():
    model_path = "models/transformer"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    weights = safetensors.torch.load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(weights)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return model, tokenizer


# =====================================================
# EXPLANATION GENERATOR
# =====================================================
def generate_explanation(text, label, implicit):
    if implicit:
        return (
            "The text does not contain explicit abusive words but expresses "
            "exclusionary or implicit hate (e.g., removal or denial of belonging). "
            "Such cases are challenging for automated systems."
        )

    if label == 1:
        return "The text contains harmful, abusive, or threatening language."

    return "The text appears non-harmful and does not express explicit hate."


# =====================================================
# OUTPUT PANEL
# =====================================================
with right_col:

    if analyze_btn:

        if not text.strip():
            st.warning("Enter text to analyze.")
            st.stop()

        # ✅ TRUE INPUT LANGUAGE
        detected_lang = detect_language_original(text)
        st.info(f"Detected Language: **{detected_lang}**")

        # --------------------
        # TRANSLATION (DISPLAY)
        # --------------------
        try:
            translated_display = GoogleTranslator(
                source="auto", target=target_code
            ).translate(text)
        except:
            translated_display = text

        st.success(f"Translated ({target_lang}): {translated_display}")

        # --------------------
        # TRANSLATE TO ENGLISH FOR MODEL
        # --------------------
        try:
            english_text = GoogleTranslator(source="auto", target="en").translate(text)
        except:
            english_text = text

        # --------------------
        # MODEL PREDICTION
        # --------------------
        model, tokenizer = load_multilingual_model()
        enc = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[0].numpy()

        # ===================================================
        # 🔴 IMPLICIT HATE PHRASES
        # ===================================================
        implicit_hate_phrases = [
            "dont belong here", "do not belong here", "they dont belong here",
            "burden on society", "drag on society",
            "remove them", "should be removed", "must be removed",
            "go back", "go back to where you came from", "send them back",
            "no place for them", "not welcome here",
            "they are the problem", "root of all problems",
            "ruining our country", "destroying our society",
            "we dont need them", "kick them out", "ban them",

            # Hinglish / Roman Hindi
            "hata dena chahiye", "inka rehna galat hai", "rehna hi galat",
            "yahan rehna galat hai", "ye log problem hain",
            "inko yahan nahi rehna chahiye",
            "wapas bhejo", "desh ke liye khatra"
        ]

        implicit_flag = any(p in english_text.lower() for p in implicit_hate_phrases)

        # Soft override for implicit hate
        if implicit_flag and probs[1] < 0.60:
            probs = np.array([0.40, 0.60])

        pred_label = int(np.argmax(probs))
        confidence = probs[pred_label] * 100

        # --------------------
        # RESULT
        # --------------------
        st.write("### Prediction Result")

        if pred_label == 1:
            st.markdown(
                f"<div class='result-box hate'>Harmful / Hate — {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box non-hate'>Non-Harmful — {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

        # --------------------
        # CONFIDENCE WARNING LEVELS
        # --------------------
        if implicit_flag:
            st.error(
                "⚠️ High Risk Content Detected\n\n"
                "This text may contain implicit or exclusionary hate. "
                "Automated predictions can be uncertain in such cases."
            )
        elif 55 <= confidence < 70:
            st.warning(
                "⚠️ Medium Confidence Prediction\n\n"
                "The model is moderately confident. Manual review is recommended."
            )
        elif confidence < 55:
            st.info(
                "ℹ️ Low Confidence Result\n\n"
                "The prediction is uncertain due to weak or ambiguous signals."
            )

        # --------------------
        # EXPLANATION
        # --------------------
        st.write("### Explanation")
        st.info(generate_explanation(english_text, pred_label, implicit_flag))
