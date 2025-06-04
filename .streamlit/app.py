import streamlit as st 

home_page= st.Page("01_home.py", title="🏠 Home")
image_page = st.Page("02_image.py", title="📸 Image")
video_page = st.Page("03_video_test.py", title="🎬 Video PPE Analysis")
live_page = st.Page("04_live.py", title="📹 Webcam")

pg = st.navigation([home_page, image_page, video_page, live_page])
# st.set_page_config(page_title="Home", page_icon="🏠")
pg.run()