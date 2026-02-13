# ui_theme.py
import base64
import streamlit as st
from pathlib import Path

def apply_dark_glass_theme(bg_image_path: str = "assets/dark_bg.jpg"):
    """
    Apply the common dark glassmorphism theme with a local background image.
    Call this once at the top of every page (after render_navbar).
    """

    # Try to read and encode the local background image
    img_path = Path(bg_image_path)
    if img_path.is_file():
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        # data URL for CSS
        bg_layer = f"url('data:image/jpeg;base64,{encoded}')"
        background_css = (
            "radial-gradient(circle at top, #1b2735 0%, #090a0f 55%, #000000 100%), "
            f"{bg_layer} center/cover fixed no-repeat"
        )
    else:
        # Fallback: only gradient if image missing
        background_css = "radial-gradient(circle at top, #1b2735 0%, #090a0f 55%, #000000 100%)"

    st.markdown(f"""
    <style>
    /* Full-page background with local image */
    body {{
        background: {background_css};
        font-family: "Segoe UI", sans-serif;
    }}

    /* Center main block a bit */
    .block-container {{
        padding-top: 1.5rem;
    }}

    /* Glass card style */
    .glass-card {{
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 22px;
    }}

    /* Titles */
    .main-title {{
        font-size: 40px;
        text-align: center;
        font-weight: 800;
        color: #e3f2fd;
        text-shadow: 0 0 12px rgba(0, 188, 212, 0.6);
        padding-top: 10px;
    }}

    .sub-title {{
        text-align: center;
        font-size: 16px;
        color: #cfd8dc;
        margin-bottom: 18px;
    }}

    /* Small rounded buttons */
    .stButton>button {{
        background-image: linear-gradient(135deg, #0288d1, #01579b);
        color: white;
        padding: 6px 14px;    /* smaller */
        border-radius: 999px; /* pill */
        border: none;
        font-size: 14px;
        font-weight: 600;
        transition: all 0.25s ease;
    }}
    .stButton>button:hover {{
        transform: scale(1.06) translateY(-1px);
        background-image: linear-gradient(135deg, #039be5, #0277bd);
    }}

    /* Navbar style (chips – your navbar can use these classes) */
    .top-nav {{
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 18px;
    }}
    .nav-chip {{
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        text-decoration: none;
        border: 1px solid rgba(255,255,255,0.28);
        color: #e0f7fa;
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
    }}
    .nav-chip:hover {{
        transform: translateY(-2px);
        box-shadow: 0 0 12px rgba(0,188,212,0.4);
    }}
    .nav-chip-active {{
        background: linear-gradient(135deg,#00bcd4,#1976d2);
        color: white;
        border-color: transparent;
        box-shadow: 0 4px 18px rgba(0,188,212,0.55);
    }}

    /* Info / success boxes text color fix on dark bg */
    .stAlert {{
        background-color: rgba(255,255,255,0.08) !important;
        color: #eceff1 !important;
    }}

    /* Markdown text color */
    .stMarkdown, .stText, .stTextInput, label, p {{
        color: #eceff1 !important;
    }}

    /* Text area styling */
    textarea {{
        background: rgba(0,0,0,0.35) !important;
        color: #e0f7fa !important;
        border-radius: 12px !important;
    }}

    /* Scrollbar subtle */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #37474f;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)
