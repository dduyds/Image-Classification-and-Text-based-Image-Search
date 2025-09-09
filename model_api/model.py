import torch
import torch.nn as nn
import numpy as np


""""
Nhúng vị trí cho các token trong mô hình Transformer.
"""
class PositionalEncoding(nn.Module):
  def __init__(self, width, max_seq_length):
    super().__init__() # Khởi tạo các thuộc tính và phương của class cha nn.Module cho class con PE

    # Không cho mô hình cập nhật tham số này nên không nhất thiết phải dùng self.
    position_encoding = torch.zeros(max_seq_length, width) # Tạo ma trận để chưa vị trí
    # print(f"Shape of position encoding: {position_encoding.shape}")
    # Duyệt qua từng token
    for element in range(max_seq_length):
      # Duyệt qua từng phần tử trong token theo chiều sâu
      for index in range(width):
        if index % 2 == 0:
          # Tính toán giá trị cho vị trí "chẵn"
          position_encoding[element, index] = np.sin(element / (10000 ** (index / width)))
        else:
          # Tính toán giá trị cho vị trí "lẽ"
          position_encoding[element, index] = np.cos(element / (10000 ** ((index - 1) / width)))

    # Đăng kí position_encoding vào mô hình "state_dict()" nhưng mô hình sẽ không cập nhật cho tham số này
    self.register_buffer('position_encoding', position_encoding.unsqueeze(0))

  def forward(self, x):
    # Cộng thêm giá trị về vị trí cho token
    x = x + self.position_encoding
    return x

"""
Lớp AttentionHead mô tả một đầu attention trong mô hình Transformer.
"""
class AttentionHead(nn.Module):
  def __init__(self, width, head_size):
    super().__init__()
    self.head_size = head_size

    self.query = nn.Linear(width, head_size)
    self.key = nn.Linear(width, head_size)
    self.value = nn.Linear(width, head_size)

  def forward(self, x, mask=None):

    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    # Nhân ma trận queries cho ma trận keys
    attention = q @ k.transpose(-1, -2) # Trước khi nhân cần đổi 2 chiều cuối của k

    # Chuẩn hóa - scaling
    attention = attention / (self.head_size **0.5)

    # Đưa mask vào attention
    if mask is not None:
      # Mask những vị trí có giá trị là 0 thành - vô cùng để đầu ra softmax là 0
      attention = attention.masked_fill(mask == 0, float("-inf"))

    # Cho ma trận đi qua softmax
    attention = torch.softmax(attention, dim=-1)

    # Tính độ tương quan giữa các token
    attention = attention @ v

    return attention

"""
Lớp MultiHeadAttention mô tả một lớp attention với nhiều đầu attention.
"""
class MultiHeadAttention(nn.Module):
  def __init__(self, width, n_heads):
    super().__init__()
    # Tính toán chiều cho mỗi head
    self.head_size = width // n_heads

    # Tạo một lớp dense để tính toán đầu ra
    self.linear = nn.Linear(width, width)

    self.heads = nn.ModuleList([AttentionHead(width, self.head_size) for _ in range(n_heads)])

  def forward(self, x, mask=None):
    # Gộp các attention heads lại đồng thời truyền x vào
    attention_heads = torch.cat([head(x, mask) for head in self.heads], dim=-1)

    # Đưa các attention heads đi qua lớp Linear, tổng hợp thông tin
    output = self.linear(attention_heads)

    return output

"""
Lớp TransformerEncoder mô tả một lớp mã hóa của mô hình Transformer.
"""
class TransformerEncoder(nn.Module):
  def __init__(self, width, n_heads, dim_expand = 4):
    super().__init__()
    self.width = width
    self.n_heads = n_heads

    # Normalization 1 Layer
    self.ln1 = nn.LayerNorm(width)

    # Multi-Head Attention Layer
    self.mha = MultiHeadAttention(width, n_heads)

    # Normalization 2 Layer
    self.ln2 = nn.LayerNorm(width)

    # Multilayer Perception Layer
    self.mlp = nn.Sequential(
        nn.Linear(self.width, self.width * dim_expand),
        nn.GELU(),
        nn.Linear(self.width * dim_expand, self.width)
      )

  def forward(self, x, mask=None):
    x = x + self.mha(self.ln1(x), mask=mask)
    x = x + self.mlp(self.ln2(x))

    return x
  
"""
Hàm Tokenizer chuyển đổi văn bản thành các token và tạo mặt nạ (mask) cho mô hình.
Đồng thời thực hiện quá trình giải mã cho token thành văn bản nếu encode = False.
"""
def Tokenizer(text, encode = True, mask = None, max_seq_length = 32):
  if encode:
    target_text = chr(2) + text + chr(3) # Thêm các token đặc biệt SOT và EOT
    target_text = target_text + "".join([chr(0) for _ in range(max_seq_length - len(target_text))]) # Thêm Padding
    target_text = torch.IntTensor(list(target_text.encode("utf-8"))) # Encoding text
    # Tạo ma trận 1 với vị trí có text
    # Tạo các giá trị 0 tại các vị trí có kí tự đặc biệt
    mask = (target_text != 0).int()
  else:
    target_text = [chr(x) for x in text[1:mask.sum().item() - 1]]
    target_text = "".join(target_text)
    mask = None
  return target_text, mask

"""
Lớp TextEncoder chuyển đổi văn bản thành các đặc trưng nhúng (embedding) sử dụng mô hình CLIP
"""
class TextEncoder(nn.Module):
  def __init__(self, vocab_size, width, max_seq_length, n_heads, n_layers, embedding_dim):
    super().__init__()

    # Giới hạn số lượng token trong 1 câu
    self.max_seq_length = max_seq_length

    # Tạo ra 1 lớp embedding để gán giá trị cho các token, token[0] = [1, width]
    self.encoder_embedding = nn.Embedding(vocab_size, width)

    # Thêm thông tin về vị trí cho từng token
    self.position_encoding = PositionalEncoding(width, max_seq_length)

    # Các đặc trưng sẽ được học sự tương tác lẫn nhau thông qua các lớp Encoder
    self.encoder_layers = nn.ModuleList([TransformerEncoder(width, n_heads) for _ in range(n_layers)])

    # Tạo ra một ma trận tham số để có thể ánh xạ vào chung không gian với đặc trưng từ ảnh
    self.projection = nn.Parameter(torch.rand(width, embedding_dim))

  def forward(self, text, mask=None):
    """
    Batch_size: số dòng dữ liệu

    """
    # Text Embedding
    x = self.encoder_embedding(text) # [Batch_size, max_seq_length, width]
    # print(x.shape)

    # Position Encoding
    x = self.position_encoding(x) # [Batch_size, max_seq_length, width]
    # print(x.shape)

    # Transformer Encoder
    for layer in self.encoder_layers:
      x = layer(x, mask=mask)
    # [Batch_size, max_seq_length, width]
    # print(x.shape)

    ### Get feature EOT
    # Lấy token gần cuối trong câu
    # x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:, 0], dim =1), 1)] # [Batch_size, width]
    ## Bước 1: Lấy kích thước batch từ x
    batch_size = x.shape[0]

    ## Bước 3: Lấy mask ban đầu cho mỗi chuỗi (chiều thứ hai của mask)
    # Giả định rằng mask ban đầu của từng chuỗi nằm ở mask[:, 0, :] hoặc mask[:, :, 0]
    # Dựa trên cách mask được tạo trong Dataset (lặp lại mask 1D để thành mask 2D cho mỗi item,
    # sau đó DataLoader xếp các item thành batch),
    # mask[i] là mask 2D (32, 32) của item i.
    # mask[:, 0] lấy cột đầu tiên của mask[i] cho mỗi item i trong batch.
    # Tức là mask[:, 0] có shape (Batch_size, max_seq_length)
    mask_default = mask[:, 0]

    ## Bước 4: Tính tổng trên chiều max_seq_length để được độ dài thực tế của mỗi chuỗi
    length_of_not_spec_token = torch.sum(mask_default, dim=1)

    ## Bước 5: Trừ đi 1 từ độ dài thực tế để có chỉ số của token cuối cùng
    # (Vì chỉ số bắt đầu từ 0)
    lastest_index = torch.sub(length_of_not_spec_token, 1)

    ## Bước 6: Sử dụng lastest_index để chọn các vector từ x
    # Lấy phần tử x[i, lastest_index[i], :] cho mỗi i từ 0 đến batch_size - 1
    x_result = x[torch.arange(batch_size), lastest_index]
    # Gán kết quả này trở lại cho biến x
    x = x_result
    # print(f"Text {x.shape}")

    # Joint miltimodel embedding
    if self.projection is not None:
      x = x @ self.projection # [Batch_size, embedding_dim]

    x = x / torch.norm(x, dim=1, keepdim=True) # [Batch_size, embedding_dim]

    return x

"""
Lớp ImageEncoder chuyển đổi hình ảnh thành các đặc trưng nhúng (embedding) sử dụng mô hình CLIP
"""
class ImageEncoder(nn.Module):
  def __init__(self, width, img_size, patch_size, n_channels, n_layers, n_heads, embedding_dim):
    super().__init__()
    # Kiểm tra xem việc chia patch size có đúng với kích thước ảnh không
    assert img_size[1] % patch_size[0] == 0 and img_size[2] % patch_size[1] == 0
    # Heads phải được chia hết bởi width để không bị lỗi
    assert width % n_heads == 0

    # Tính toán số lượng patch
    self.n_patches = (img_size[1] * img_size[2]) // (patch_size[0] * patch_size[1])

    # Thêm vị trí cho classification token (đây là 1 kỹ thuật của ViT)
    self.max_seq_length = self.n_patches + 1

    # Tích chập cho các patch trong hình ảnh
    # self.patch_embedding = nn.Conv3d(n_channels, width, kernel_size=patch_size, stride=patch_size)
    self.patch_embedding = nn.Conv2d(n_channels, width, kernel_size=patch_size, stride=patch_size)

    # Classification token đại diện cho việc học thông tin của các patch khác,
    # nó đại diện cho toàn ảnh
    self.classification_token = nn.Parameter(torch.randn(1, 1, width))

    # Các đặc trưng trích xuất từ patch sẽ được thêm thông tin về vị trí để tránh bị xáo trộn
    self.positional_encoding = PositionalEncoding(width, self.max_seq_length)

    # Transformer encoder cho phép các đặc trưng học được sự tương đồng lần nhau
    self.encoder = nn.ModuleList([TransformerEncoder(width, n_heads) for _ in range(n_layers)])

    # Tạo các parameters thành một ma trận để có thể ánh xạ đặc trưng ảnh vào không gian
    # cùng chiều với đặc trưng từ text
    self.projection = nn.Parameter(torch.rand(width, embedding_dim))

  def forward(self, x):
    """
    Batch_size: số dòng dữ liệu
    width: chiều của mô hình image encoder
    H_p = image_size[0]//patch_size[0]: chiều cao của ảnh sau khi chia thành patch
    W_p = image_size[1]//patch_size[1]: chiều rộng của ảnh sau khi chia thành patch
    T_p = H_p + W_p
    """
    # Patch Embedding
    x = self.patch_embedding(x) # [Batch_size, width, H_p, W_p]
    # print(x.shape)
    x = x.flatten(2).transpose(1, 2) # [Batch_size, width, H_p + W_p] -> [Batch_size, T_p, width]
    
    # print(x.shape)

    # Add classification token
    x = torch.cat((self.classification_token.expand(x.size()[0], -1, -1),x), dim=1) # [Batch_size, T_p + 1, width]
    # print(x.shape)

    # Postion Encoding
    x = self.positional_encoding(x) # [Batch_size, T_p + 1, width]

    # Transformer Encoder
    for encoder_layer in self.encoder:
      x = encoder_layer(x)
    # [Batch_size, T_p + 1, width]

    # Get classification token for projection
    x = x[:, 0] # [Batch_size, width]
    # print(f"Image {x.shape}")

    # Project classification token to new dim
    if self.projection is not None:
      x = x @ self.projection # [Batch_size, embedding_size]


    # Normalize output token
    x = x / torch.norm(x, dim=1, keepdim=True) # [Batch_size, embedding_size]
    # print(x.shape)
    # print('\n')

    return x

"""
ClipModel kết hợp ImageEncoder và TextEncoder để tạo ra mô hình CLIP.
"""
class ClipModel(nn.Module):
  def __init__(self, emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads,
               vocab_size, text_width, max_seq_length, text_heads, text_layers):
    super().__init__()

    self.image_encoder = ImageEncoder(vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, emb_dim)

    self.text_encoder = TextEncoder(vocab_size, text_width, max_seq_length, text_heads, text_layers, emb_dim)

    # khởi tạo và học tham số temperature cho hàm similarity / contrastive loss
    self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

  def forward(self, image, text, mask=None):
    # Image Encoder
    image_encoder = self.image_encoder(image)

    # Text Encoder
    text_encoder = self.text_encoder(text, mask = mask)
    # print(image_encoder.shape)
    # print(text_encoder.shape)

    logits = (image_encoder @ text_encoder.transpose(-2, -1)) * torch.exp(self.temperature)
    # print(f"Logits {logits.shape}")

    ## Tính toán mất mát đối xứng
    labels = torch.arange(logits.shape[0], device=logits.device)
    # print(f"Labels {labels.shape}")
    # Mất mát image so với text
    loss_image = nn.functional.cross_entropy(logits, labels)
    # Mất mát text so với image
    loss_text = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)

    loss = (loss_image + loss_text)/2

    return loss
  









