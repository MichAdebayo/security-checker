import streamlit as st 
import os

# Fix the logo path - use absolute path from current file location
logo_path = os.path.join(os.path.dirname(__file__), "..", "images", "logo.webp")

if os.path.exists(logo_path):
    st.image(logo_path)
else:
    st.info("🛡️ Smart Safety Monitor")

st.html(
    "<h1 style='text-align:center'>Respect des règles. Garantie par l'image.</h1>"
    )