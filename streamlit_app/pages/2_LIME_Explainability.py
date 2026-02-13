import streamlit as st
import torch
import numpy as np
import safetensors.torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from navbar import render_navbar

# PAGE CONFIG
st.set_page_config(page_title="LIME Explainability", page_icon="🧪", layout="wide")
render_navbar(active="lime")

st.markdown("<h1 class='main-title'>🧪 LIME Explainability</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Understand why the model predicted harmful / non-harmful by analyzing word contributions.</p>", unsafe_allow_html=True)

# MODEL
@st.cache_resource
def load_multilingual_model():
    model_path = "models/transformer"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    weights = safetensors.torch.load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(weights)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return model, tokenizer

model, tokenizer = load_multilingual_model()

# ====================================================
#       TWO COLUMN LAYOUT (INPUT LEFT / OUTPUT RIGHT)
# ====================================================
left_col, right_col = st.columns([1.1, 1])

# ---------------------------
# LEFT → INPUT
# ---------------------------
with left_col:

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    text = st.text_area("Enter text to analyze using LIME:", height=260)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    generate = st.button("Generate LIME Explanation", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# RIGHT → OUTPUT
# ---------------------------
with right_col:

    if generate:

        if not text.strip():
            st.warning("Please enter text before generating LIME output.")
            st.stop()

        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.write("### 📊 Word Contribution Analysis")

        def predict_fn(texts):
            enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=1).numpy()
            return probs

        explainer = LimeTextExplainer(class_names=["Non-Harmful", "Harmful"])

        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=10,
            top_labels=1
        )

        lime_html = exp.as_html()
        st.components.v1.html(lime_html, height=900, scrolling=True)

        st.markdown("</div>", unsafe_allow_html=True)
