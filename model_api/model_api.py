# api_clip.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader
from model import ClipModel, Tokenizer
from dataset import Cifar10

# --- CONFIG ---v
MODEL_PATH = "./model/clip_model_cifa10.pt"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
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

# --- LOAD MODEL & DATASET ON STARTUP ---
model = ClipModel(
    emb_dim, vit_width, img_size, patch_size, n_channels,
    vit_layers, vit_heads, vocab_size, text_width,
    max_seq_length, text_heads, text_layers
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = Cifar10(train=False)

app = FastAPI()

# --- SCHEMAS ---
class PredictRequest(BaseModel):
    idx: int
    topk: int = 5

class SearchRequest(BaseModel):
    caption: str
    top_k: int = 2

# --- ENDPOINT: CLASS PREDICTION ---
@app.post("/predict")
def predict(req: PredictRequest):
    tokenizer = Tokenizer
    text_tokens = torch.stack([tokenizer(x)[0] for x in CLASS_NAMES]).to(DEVICE)
    mask = torch.stack([tokenizer(x)[1] for x in CLASS_NAMES])
    mask = mask.repeat(1, len(mask[0])).reshape(len(mask), len(mask[0]), len(mask[0])).to(DEVICE)

    sample = dataset[req.idx]
    image = sample["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.image_encoder(image)
        text_features = model.text_encoder(text_tokens, mask=mask)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(req.topk)

    predictions = [(CLASS_NAMES[int(i)], float(v) * 100) for v, i in zip(values, indices)]
    decoded_caption = Tokenizer(
        dataset[req.idx]["caption"], encode=False, mask=dataset[req.idx]["mask"][0]
    )[0]

    return {
        "predictions": predictions,
        "caption": decoded_caption,
        "image_tensor": sample["image"].tolist()
    }

# --- ENDPOINT: CAPTION SEARCH ---
@app.post("/search")
def search(req: SearchRequest):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    token, mask = Tokenizer(req.caption)
    token = token.unsqueeze(0).to(DEVICE)
    mask = mask.unsqueeze(0).to(DEVICE)
    mask = mask.repeat(1, mask.shape[1]).reshape(1, mask.shape[1], mask.shape[1])

    with torch.no_grad():
        text_feature = model.text_encoder(token, mask=mask.to(DEVICE))
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

    image_features = []
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(DEVICE)
            feats = model.image_encoder(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            image_features.append(feats)

    image_features = torch.cat(image_features, dim=0)
    similarity = (100.0 * text_feature @ image_features.T).squeeze(0)
    values, indices = similarity.topk(req.top_k)

    results = []
    for i, v in zip(indices, values):
        sample = dataset[int(i)]
        results.append({
            "idx": int(i),
            "similarity": float(v),
            "caption_tokens": sample["caption"],
            "mask": sample.get("mask", None).tolist() if sample.get("mask", None) is not None else None,
            "image": sample["image"].tolist(),
        })

    return {"results": results}
