import streamlit as st 

home_page= st.Page("01_home.py",title="🏠 Home")
cam_page = st.Page("02_vision.py",title="📸 Vision Computer")

pg = st.navigation([home_page, cam_page])
st.set_page_config(page_title="Home", page_icon="🏠")
pg.run()