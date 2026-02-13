import streamlit as st
from navbar import render_navbar

st.set_page_config(page_title="Home", layout="wide", page_icon="🏠")
render_navbar(active="home")

# ===============================
# MODERN HEADER
# ===============================
st.markdown("""
<style>
.home-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #0066ff, #00bbff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
    margin-bottom: 0px;
}

.home-subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
    margin-bottom: 30px;
}

.feature-card {
    background: white;
    border-radius: 18px;
    padding: 25px;
    border: 1px solid #e6e6e6;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.08);
    transition: 0.2s;
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0px 10px 28px rgba(0,0,0,0.15);
}

.feature-icon {
    font-size: 42px;
    margin-bottom: 12px;
}

footer {
    text-align: center;
    margin-top: 40px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='home-title'>Multilingual Hate & Harmful Speech Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='home-subtitle'>A system for multilingual toxic content detection, harm analysis & explanations.</p>", unsafe_allow_html=True)

# ===============================
# FEATURE CARDS (Modern & Clean)
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>💬</div>
        <h3>What is Hate Speech?</h3>
        <p>
        Hate speech refers to abusive, insulting, or threatening expressions directed at an individual 
        or group based on religion, caste, gender, ethnicity, nationality, or identity. It can escalate conflict,
        spread discrimination, and cause real-world harm.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>🧠</div>
        <h3>What the System Does</h3>
        <p>
        The system detects harmful or toxic content using advanced multilingual Transformer models (XLM-R).
        It predicts whether text is harmful and explains why using AI explainability tools like LIME.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>⚖️</div>
        <h3>Why Harmful vs Non-Harmful?</h3>
        <p>
        Distinguishing harmful content is essential for online safety, responsible moderation, and preventing
        harassment. It protects communities and ensures healthy digital communication.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<footer>
    Use the navigation above to test detection, view LIME explanations, and explore features.
</footer>
""", unsafe_allow_html=True)
