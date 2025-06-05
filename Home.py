import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="centered"
)

# Main title
st.title("📦 CLIP Application on CIFAR-10")
st.markdown("---")

# Introduction
st.markdown("""
## 👋 Introduction

Welcome to the demo application of a **custom CLIP model** trained on the **CIFAR-10** dataset!

This app allows you to:
- 🏷️ **Classify images** from the CIFAR-10 test set with 10 basic categories.
- 🔍 **Search for similar images** based on textual descriptions (captions) using **similarity** between embeddings.

---

## 🔧 Technologies Used
- [x] Build and custom CLIP model (PyTorch)
- [x] CIFAR-10 dataset (10,000 test images)
- [x] Lightweight, user-friendly UI with Streamlit

---

## 📂 Navigation
Use the **left sidebar** to access different functionalities:
- 👉 **Image Classification**
- 👉 **Image Search by Caption**

---

## 📘 About CLIP
CLIP (Contrastive Language–Image Pretraining) is a model that learns joint embeddings of images and text into a **shared vector space**, enabling cross-modal retrieval between images and captions.
""")

# Footer
st.markdown("---")