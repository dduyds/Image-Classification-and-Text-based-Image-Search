# Home.py
import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="centered"
)

st.title("📦 CLIP Application on CIFAR-10")
st.markdown("---")

st.markdown("""
## 👋 Introduction

Welcome to the demo application of a **custom CLIP model** trained on the **CIFAR-10** dataset!

This app allows you to:
- 🏷️ **Classify images** from the CIFAR-10 test set with 10 basic categories.
- 🔍 **Search for similar images** based on textual descriptions (captions) using **similarity** between embeddings.

---

## 🔧 Technologies Used
- [x] Build and custom CLIP model (PyTorch) from scratch            
- [x] FastAPI backend for model inference
- [x] Streamlit frontend


---

## 📂 Navigation
Use the **left sidebar** to access:
- 👉 Image Classification
- 👉 Image Search by Caption
""")
