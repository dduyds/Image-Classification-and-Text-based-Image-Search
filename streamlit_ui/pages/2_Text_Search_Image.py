# pages/2_Text_Search_Image.py
import streamlit as st
import requests
import torch
import os

# API_URL = "http://localhost:8000/search"
# Must get API_BASE from env (this is declare in docker-compose file)
API_BASE = os.getenv("API_URL", "http://localhost:8000")
API_URL = f"{API_BASE}/search"

st.set_page_config(page_title="Find Similar Images", layout="centered")
st.title("🔎 Find Similar Images by Caption")

caption = st.text_input("Enter a caption (e.g., 'an image of dog')", value="an image of dog")
top_k = st.slider("Select number of top-k similar images", 1, 4, value=2)

if st.button("🔍 Search"):
    with st.spinner("⏳ Calling API..."):
        try:
            response = requests.post(API_URL, json={"caption": caption, "top_k": top_k})
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.stop()

    st.markdown(f"### 📌 Search Results for: *\"{caption}\"*")
    cols = st.columns(top_k)

    for i, result in enumerate(data["results"]):
        with cols[i]:
            image_tensor = torch.tensor(result["image"])
            image = image_tensor.permute(1, 2, 0).numpy()
            st.image(image, caption=f"Similarity: {result['similarity']:.2f}%", use_container_width=True)
