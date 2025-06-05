import streamlit as st
import torch
from model import ClipModel, Tokenizer
from dataset import Cifar10
from torchvision.transforms.functional import to_pil_image

# --- CONFIG ---
MODEL_PATH = "./model/clip_model_cifa10.pt"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL PARAMETERS ---
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

# --- LOAD DATASET ---
@st.cache_data
def load_dataset():
    return Cifar10(train=False)

# --- INFERENCE ---
def predict_topk(model, dataset, idx, topk=5):
    tokenizer = Tokenizer
    text_tokens = torch.stack([tokenizer(x)[0] for x in CLASS_NAMES]).to(DEVICE)
    mask = torch.stack([tokenizer(x)[1] for x in CLASS_NAMES])
    mask = mask.repeat(1, len(mask[0])).reshape(len(mask), len(mask[0]), len(mask[0])).to(DEVICE)

    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.image_encoder(image)
        text_features = model.text_encoder(text_tokens, mask=mask)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(topk)

    predictions = [(CLASS_NAMES[int(i)], float(v) * 100) for v, i in zip(values, indices)]
    return sample["image"], predictions

# --- STREAMLIT UI ---
st.set_page_config(page_title="CLIP Inference", layout="centered")
st.title("🔍 CLIP Inference on CIFAR-10")

idx = st.number_input("🔢 Enter image index (0–9999):", min_value=0, max_value=9999, value=0, step=1)
topk = st.slider("🔝 Number of top-k predictions:", min_value=1, max_value=10, value=5)

if st.button("🚀 Predict"):
    with st.spinner("⏳ Loading model and data..."):
        model = load_model()
        dataset = load_dataset()

    st.write("📷 Processing image and caption...")
    with st.spinner("🤖 Running inference..."):
        image_tensor, predictions = predict_topk(model, dataset, idx, topk)

        decoded_caption = Tokenizer(
            dataset[idx]["caption"], encode=False, mask=dataset[idx]["mask"][0]
        )[0]

    # st.image(
    #     to_pil_image(image_tensor),
    #     caption=f"📝 Original caption: {decoded_caption}",
    #     # use_container_width=True
    #     width=128
    # )

    # Resize to a suitable size
    image = to_pil_image(image_tensor).resize((640, 640))

    # Create 3 layout to put the image in the middle 
    cols = st.columns([1, 2, 1])
    with cols[1]:  # middle
        st.image(image, caption=f"📝 Original caption: {decoded_caption}", use_container_width=False)

    st.markdown("### 📌 Predictions:")
    for label, prob in predictions:
        st.write(f"- **{label}**: {prob:.2f}%")