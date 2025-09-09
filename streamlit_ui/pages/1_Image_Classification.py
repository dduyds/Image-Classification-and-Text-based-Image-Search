# pages/1_Image_Classification.py
import streamlit as st
import requests
import torch
import os
from torchvision.transforms.functional import to_pil_image

# API_URL = "http://localhost:8000/predict"
API_BASE = os.getenv("API_URL", "http://localhost:8000")
API_URL = f"{API_BASE}/predict"

st.set_page_config(page_title="CLIP Inference", layout="centered")
st.title("🏷️ Classify image by Index")

idx = st.number_input("🔢 Enter image index (0–9999):", min_value=0, max_value=9999, value=0, step=1)
topk = st.slider("🔝 Number of top-k predictions:", min_value=1, max_value=10, value=5)

if st.button("🚀 Predict"):
    with st.spinner("⏳ Calling API..."):
        try:
            response = requests.post(API_URL, json={"idx": idx, "topk": topk})
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.stop()

    # Convert image tensor back to PIL
    image_tensor = torch.tensor(data["image_tensor"])
    image = to_pil_image(image_tensor).resize((640, 640))

    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.image(image, caption=f"📝 Original caption: {data['caption']}", use_container_width=False)

    st.markdown("### 📌 Predictions:")
    for label, prob in data["predictions"]:
        st.write(f"- **{label}**: {prob:.2f}%")
