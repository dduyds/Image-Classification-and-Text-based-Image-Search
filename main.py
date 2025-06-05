from model import Tokenizer, ClipModel
from dataset import Cifar10
from torch.utils.data import DataLoader
import argparse
import torch
from model_mode import train_clip_model, evaluate_clip_model, predict_and_show_topk

parser = argparse.ArgumentParser(description="Chọn chế độ chạy chương trình: TRAIN, TEST, INFERENCE")
parser.add_argument("--mode", type=str, default="INFERENCE", choices=["TRAIN", "TEST", "INFERENCE"], help="Chế độ chạy: TRAIN / TEST / INFERENCE")
# Tham số tùy chọn chỉ áp dụng cho INFERENCE
parser.add_argument("--idx", type=int, default=0, help="Chỉ số ảnh cần dự đoán (chỉ dùng cho chế độ INFERENCE)")

args = parser.parse_args()
# Mode chạy chương trình
mode = args.mode.upper()
# Index ảnh để INFERENCE
idx = args.idx


# Khởi tạo tham số cho mô hình và dữ liệu
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

# Khởi tạo mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClipModel(emb_dim, vit_width, img_size, patch_size, n_channels,
                    vit_layers, vit_heads, vocab_size, text_width,
                    max_seq_length, text_heads, text_layers).to(device)

if __name__ == "__main__":
    if mode == "TRAIN":
        # Tạo DataLoader cho tập dữ liệu Cifar10
        cifa10_train_dataset = Cifar10(train=True)
        cifa10_train_loader = DataLoader(cifa10_train_dataset, batch_size=batch_size, shuffle=True)

        # Huấn luyện mô hình CLIP
        train_clip_model(
            model=model,
            train_loader=cifa10_train_loader,
            epochs=epochs,
            lr=lr,
            save_path="./model/clip_model_cifa10.pt",
            device=device
        )
    elif mode == "TEST":
        # Tạo DataLoader cho tập dữ liệu Cifar10
        cifa10_test_dataset = Cifar10(train=False)
        cifa10_test_loader = DataLoader(cifa10_test_dataset, batch_size=batch_size, shuffle=False)

        # Đánh giá mô hình CLIP
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
            idx=idx,  # Chỉ số ảnh để kiểm tra
            device=device,
            topk=5
        )