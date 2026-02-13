import streamlit as st

def render_navbar(active="home"):

    # ================= GLOBAL UI + BACKGROUND (Blue Modern) ================= #
    st.markdown("""
    <style>

    /* REMOVE STREAMLIT DEFAULT TOP HEADER */
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* REMOVE SIDEBAR COMPLETELY */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    /* PAGE BACKGROUND */
    body {
        background: #F5F8FF !important;
    }

    /* MAIN CONTAINER (white clean box) */
    .block-container {
        background: rgba(255,255,255,0.85) !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }

    /* ======================= NAVBAR ======================= */
    .custom-navbar {
        width: 100%;
        padding: 16px;
        text-align: center;
        position: sticky;
        top: 0;
        z-index: 1000;
        background: #1A73E8;
        border-radius: 0 0 14px 14px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.15);
    }

    .nav-btn {
        display: inline-block;
        margin: 0 25px;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 17px;
        text-decoration: none;
        border-radius: 10px;
        color: white !important;
        transition: 0.25s ease;
    }

    .nav-btn:hover {
        background: rgba(255,255,255,0.18);
        transform: translateY(-2px);
    }

    .active-nav {
        background: white !important;
        color: #1A73E8 !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }

    /* ======================= CARD ======================= */
    .card {
        background: white;
        border-left: 5px solid #1A73E8;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        transition: 0.25s ease;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 26px rgba(0,0,0,0.15);
    }

    /* ======================= TEXT COLORS ======================= */
    .main-title {
        color: #1A73E8 !important;
        text-align: center;
        font-size: 42px;
        font-weight: 800;
    }

    .sub-title {
        text-align: center;
        color: #5f6368 !important;
        font-size: 18px;
        margin-bottom: 25px;
    }

    label, p, h3 {
        color: #1A1A1A !important;
    }

    /* ======================= INPUTS ======================= */
    textarea, input, .stTextInput > div > div > input {
        background-color: #E8F0FE !important;
        border: 2px solid #1A73E8 !important;
        border-radius: 12px !important;
        color: #1A1A1A !important;
    }

    /* ======================= DROPDOWN ======================= */
    .stSelectbox > div > div {
        background-color: #E8F0FE !important;
        border: 2px solid #1A73E8 !important;
        border-radius: 12px !important;
    }

    /* ======================= BUTTONS ======================= */
    .stButton > button {
        background-color: #1A73E8 !important;
        color: white !important;
        font-weight: 600;
        border-radius: 12px;
        padding: 10px 20px;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background-color: #0F5BC8 !important;
        transform: scale(1.03);
    }

    /* ======================= RESULT BOX ======================= */
    .result-box {
        padding: 18px;
        border-radius: 12px;
        font-weight: 600;
        color: white !important;
    }

    .hate { background: rgba(255, 82, 82, 0.85); }
    .non-hate { background: rgba(76, 175, 80, 0.85); }

    </style>
    """, unsafe_allow_html=True)

    # ======================= NAVBAR HTML ======================= #
    navbar_html = f"""
    <div class="custom-navbar">
        <a class="nav-btn {'active-nav' if active=='home' else ''}" href="/Home">🏠 Home</a>
        <a class="nav-btn {'active-nav' if active=='detect' else ''}" href="/Translation_and_Detection">🔍 Detection</a>
        <a class="nav-btn {'active-nav' if active=='lime' else ''}" href="/LIME_Explainability">🧪 LIME</a>
        <a class="nav-btn {'active-nav' if active=='about' else ''}" href="/About">ℹ️ About</a>
    </div>
    """

    st.markdown(navbar_html, unsafe_allow_html=True)
