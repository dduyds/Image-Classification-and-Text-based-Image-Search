from model import Tokenizer, ClipModel
from dataset import Cifar10
from torch.utils.data import DataLoader
import argparse
import torch
from model_mode import train_clip_model, evaluate_clip_model, predict_and_show_topk

parser = argparse.ArgumentParser(description="Chọn chế độ chạy chương trình: TRAIN, TEST, INFERENCE")
parser.add_argument("--mode", type=str, default="INFERENCE", choices=["TRAIN", "TEST", "INFERENCE"], help="Chế độ chạy: TRAIN / TEST / INFERENCE")
# Optional param for INFERENCE mode
parser.add_argument("--idx", type=int, default=0, help="Chỉ số ảnh cần dự đoán (chỉ dùng cho chế độ INFERENCE)")

args = parser.parse_args()
mode = args.mode.upper()
idx = args.idx


# Initiate model params
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
lr = 1e-3
epochs = 10
batch_size = 128

# Initiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClipModel(emb_dim, vit_width, img_size, patch_size, n_channels,
                    vit_layers, vit_heads, vocab_size, text_width,
                    max_seq_length, text_heads, text_layers).to(device)

if __name__ == "__main__":
    if mode == "TRAIN":
        # Create Train DataLoader for Cifar10 dataset
        cifa10_train_dataset = Cifar10(train=True)
        cifa10_train_loader = DataLoader(cifa10_train_dataset, batch_size=batch_size, shuffle=True)

        # Train CLIP model
        train_clip_model(
            model=model,
            train_loader=cifa10_train_loader,
            epochs=epochs,
            lr=lr,
            save_path="./model/clip_model_cifa10.pt",
            device=device
        )
    elif mode == "TEST":
        # Create Test DataLoader for Cifar10 dataset
        cifa10_test_dataset = Cifar10(train=False)
        cifa10_test_loader = DataLoader(cifa10_test_dataset, batch_size=batch_size, shuffle=False)

        # Evaluate CLIP model
        accuracy = evaluate_clip_model(
            model_path="./model/clip_model_cifa10.pt",
            model_init=model,
            tokenizer=Tokenizer,
            dataset=cifa10_test_dataset,
            dataloader=cifa10_test_loader,
            device=device
        )
        print(f"\nModel Accuracy: {accuracy} %")
    elif mode == "INFERENCE":
        predict_and_show_topk(
            model_path="./model/clip_model_cifa10.pt",
            model_init=model,
            tokenizer=Tokenizer,
            dataset=Cifar10(train=False),
            idx=idx,  # Index of image for prediction
            device=device,
            topk=5
        )