import streamlit as st
from navbar import render_navbar

st.set_page_config(page_title="About", layout="wide", page_icon="ℹ️")
render_navbar(active="about")

st.markdown("<h1 class='main-title'>ℹ️ About This Project</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Details about the system, technology stack, and capabilities.</p>", unsafe_allow_html=True)

st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
st.write("""
### 📌 Overview  
This project performs multilingual hate & harmful speech detection using advanced AI models.

### 🔍 Key Capabilities  
- Harmful vs Non-Harmful detection  
- Hate vs Non-hate analysis  
- Multilingual transformer model  
- Kannada specialized models  
- LIME explainability  
- Automatic translation  
""")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
st.write("""
### 🛠 Technologies Used  
- Python  
- Streamlit  
- PyTorch  
- Transformers  
- Scikit-learn  
- Deep Translator  
- LIME Explainability  
""")
st.markdown("</div>", unsafe_allow_html=True)
