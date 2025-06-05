import streamlit as st
import torch
from model import ClipModel, Tokenizer
from dataset import Cifar10
from torch.utils.data import DataLoader

# --- CONFIG ---
MODEL_PATH = "./model/clip_model_cifa10.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2000

# --- MODEL PARAMS ---
emb_dim = 64
vit_width = 64
img_size = (3, 32, 32)
patch_size = (4, 4)
n_channels = 3
vit_layers = 4
vit_heads = 8
vocab_size = 256
text_width = 32
max_seq_length = 32
text_heads = 8
text_layers = 4

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = ClipModel(
        emb_dim, vit_width, img_size, patch_size, n_channels,
        vit_layers, vit_heads, vocab_size, text_width,
        max_seq_length, text_heads, text_layers
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# --- SEARCH FUNCTION ---
def search_similar_images(caption, model, dataset, tokenizer, top_k=5, batch_size=2000):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Encode caption
    token, mask = tokenizer(caption)
    token = token.unsqueeze(0).to(DEVICE)
    mask = mask.unsqueeze(0).to(DEVICE)
    mask = mask.repeat(1, mask.shape[1]).reshape(1, mask.shape[1], mask.shape[1])

    with torch.no_grad():
        text_feature = model.text_encoder(token, mask=mask.to(DEVICE))
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

    image_features = []
    all_images = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(DEVICE)
            feats = model.image_encoder(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)

            image_features.append(feats)
            all_images.append(imgs.cpu())

    image_features = torch.cat(image_features, dim=0)  # [N, D]
    all_images = torch.cat(all_images, dim=0)          # [N, C, H, W]

    similarity = (100.0 * text_feature @ image_features.T).squeeze(0)
    values, indices = similarity.topk(top_k)

    indices = [int(i) for i in indices]
    values = [float(v) for v in values]
    return (indices, values)

    # return [(all_images[idx], dataset[idx]["caption"], float(score)) for idx, score in zip(indices, values)]

# --- STREAMLIT UI ---
st.set_page_config(page_title="Find Similar Images", layout="centered")
st.title("🔎 Find Similar Images by Caption")

caption = st.text_input("Enter a caption to search (e.g., 'image of dog')", value="image of dog")
top_k = st.slider("Select number of top-k similar images", 1, 4, value=2)

if st.button("🔍 Search"):
    with st.spinner("⏳ Loading model and data..."):
        model = load_model()
    dataset = Cifar10(train=False)

    results = search_similar_images(
        caption=caption,
        model=model,
        dataset=dataset,
        tokenizer=Tokenizer,
        top_k=top_k,
        batch_size=BATCH_SIZE
    )

    st.markdown(f"### 📌 Search Results for: *\"{caption}\"*")
    cols = st.columns(top_k)

    for i, (idx, sim_score) in enumerate(zip(results[0], results[1])):
        with cols[i]:
            sample = dataset[idx]
            mask = sample.get("mask", None)
            image = sample["image"].permute(1, 2, 0).numpy()  # [C, H, W] → [H, W, C]
            decoded_caption = Tokenizer(sample["caption"], encode=False, mask=mask[0] if mask is not None else None)[0]
            st.image(image, caption=f"{decoded_caption}\nSimilarity: {sim_score:.2f}%", use_container_width=True)