import streamlit as st

def load_light_theme():
    st.markdown("""
    <style>

    /* ---------- GLOBAL PAGE ---------- */
    body, .block-container {
        background: #f7f7f9 !important;
        font-family: 'Segoe UI', sans-serif;
        color: #222 !important;
        padding-top: 1rem !important;
    }

    /* ---------- FIX: renamed card → custom-card ---------- */
    .custom-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 24px;
        border: 1px solid #e6e6e6;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    /* ---------- TITLES ---------- */
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        color: #222;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 12px;
    }

    /* ---------- INPUT ---------- */
    textarea, input, select {
        background: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 10px !important;
        padding: 12px !important;
        color: #111 !important;
    }

    /* ---------- BUTTONS ---------- */
    .stButton>button {
        background: #005ae0 !important;
        color: white !important;
        font-size: 16px !important;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: #0047b3 !important;
        transform: scale(1.02);
    }

    /* ---------- RESULT BOXES ---------- */
    .good-box {
        background: #e9f7ee;
        color: #1f7a3b;
        padding: 14px;
        border-radius: 10px;
        font-weight: 600;
    }
    .bad-box {
        background: #fdecea;
        color: #d93025;
        padding: 14px;
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """, unsafe_allow_html=True)
