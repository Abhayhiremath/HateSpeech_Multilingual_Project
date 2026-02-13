# streamlit_app/components/prediction_block.py
import streamlit as st
def render(pred, conf, expl):
    st.subheader(f"Prediction: {'Harmful' if pred==1 else 'Non-Harmful'} ({conf:.2f})")
    if expl:
        st.caption("Top words influencing decision")
        st.write(expl)
