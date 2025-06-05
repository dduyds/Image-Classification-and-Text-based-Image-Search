import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import os

def train_clip_model(model, train_loader, epochs, lr, save_path, device):
    """
    Huấn luyện mô hình CLIP trên tập dữ liệu huấn luyện.

    Args:
        model (nn.Module): Mô hình CLIP đã được khởi tạo.
        train_loader (DataLoader): DataLoader cho tập train.
        epochs (int): Số epoch huấn luyện.
        lr (float): Learning rate.
        save_path (str): Đường dẫn lưu mô hình tốt nhất.
        device (torch.device): Thiết bị huấn luyện (CPU/GPU).
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            img = data["image"].to(device)
            cap = data["caption"].to(device)
            mask = data["mask"].to(device)

            # Forward và tính loss
            loss = model(img, cap, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

        # Lưu model nếu loss cải thiện
        if avg_loss <= best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved at epoch {epoch+1} with loss {avg_loss:.4f}")


def evaluate_clip_model(model_path, model_init, tokenizer, dataset, dataloader, device):
    """
    Đánh giá mô hình CLIP theo cách so sánh caption tokenized.

    Parameters:
        model_path (str): đường dẫn model đã lưu (.pt)
        model_init (nn.Module): mô hình đã được khởi tạo sẵn
        tokenizer (function): hàm Tokenizer trả về (tokens, mask)
        dataset: tập test có thuộc tính captions (dict[int -> str])
        dataloader: dataloader test có keys 'image' và 'caption'
        device: thiết bị torch.device

    Returns:
        float: accuracy (%)
    """
    model = model_init.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Chuẩn bị tokenized text captions
    text = torch.stack([tokenizer(x)[0] for x in dataset.captions.values()]).to(device)
    mask = torch.stack([tokenizer(x)[1] for x in dataset.captions.values()])
    mask = mask.repeat(1, len(mask[0])).reshape(len(mask), len(mask[0]), len(mask[0])).to(device)

    correct, total = 0, 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data["image"].to(device), data["caption"].to(device)
            image_features = model.image_encoder(images)
            text_features = model.text_encoder(text, mask=mask)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, indices = torch.max(similarity, 1)

            pred = torch.stack([tokenizer(dataset.captions[int(i)])[0] for i in indices]).to(device)
            correct += int(sum(torch.sum((pred == labels), dim=1) // len(pred[0])))
            total += len(labels)

    accuracy = 100 * correct // total
    return accuracy


def predict_and_show_topk(
    model_path,
    model_init,
    tokenizer,
    dataset,
    idx,
    device,
    topk=5
):
    """
    Dự đoán top-k caption/class cho một ảnh trong dataset và hiển thị hình ảnh kèm kết quả.

    Parameters:
        model_path (str): Đường dẫn model đã lưu (.pt)
        model_init (nn.Module): Mô hình đã được khởi tạo sẵn
        tokenizer (function): Hàm Tokenizer(text) -> (token_tensor, mask_tensor)
        class_names (List[str]): Danh sách caption/class tương ứng với từng label (index)
        dataset (Dataset): Dataset chứa ảnh và caption
        idx (int): Chỉ số ảnh trong dataset để kiểm tra
        device (torch.device): CPU hoặc CUDA
        topk (int): Số lượng caption muốn hiển thị (mặc định = 5)
    """
    
    # Load model
    model = model_init.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_names =["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

    # Tokenize all class captions
    text_tokens = torch.stack([tokenizer(x)[0] for x in class_names]).to(device)
    mask = torch.stack([tokenizer(x)[1] for x in class_names])
    mask = mask.repeat(1, len(mask[0])).reshape(len(mask), len(mask[0]), len(mask[0])).to(device)

    # Lấy ảnh từ dataset và chuyển sang batch [1, 3, H, W]
    image = dataset[idx]["image"][None, :]
    caption_token = dataset[idx]["caption"]
    mask_token = dataset[idx]["mask"][0]

    # Hiển thị ảnh và caption thật
    plt.imshow(image[0].permute(1, 2, 0))
    decoded_caption = tokenizer(caption_token, encode=False, mask=mask_token)[0]
    plt.title(decoded_caption)
    plt.axis("off")
    plt.show()

    # Dự đoán
    with torch.no_grad():
        image_features = model.image_encoder(image.to(device))
        text_features = model.text_encoder(text_tokens, mask=mask)

    # Chuẩn hóa và tính độ tương đồng
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Top-k kết quả
    values, indices = similarity[0].topk(topk)

    print(f"\nTop-{topk} predictions {decoded_caption}:\n")
    for value, index in zip(values, indices):
        print(f"{class_names[int(index)]:>16s}: {100 * value.item():.2f}%")









