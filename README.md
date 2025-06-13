# Image Classification and Text based Image Search
## Overview
This project is an interactive application that leverages a build and custom [CLIP](https://github.com/openai/CLIP) (Contrastive Language–Image Pretraining) model to **classify images** from the CIFAR-10 dataset and **retrieve similar images based on text** descriptions. The model is built and customized using PyTorch to work effectively on the [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10) dataset.

![Model Architecture](model/Architecture.jpg)

## Key Features
- Custom CLIP model architecture tailored for CIFAR-10 dataset classification and retrieval.
- Image classification and text-based image retrieval by computing and comparing embeddings.
- Clean and user-friendly interface designed with Streamlit for easy and seamless interaction.

## Demo
A demo of the application allows users to upload or select an image from CIFAR-10, classify it, and perform text-based image search to find visually similar images based on textual input.
- Click the Video below for full demo.

[![Video Demo](https://img.youtube.com/vi/F9J-cTw5CmA/hqdefault.jpg)](https://youtu.be/F9J-cTw5CmA)

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
### Option 1. Run the Python script directly with modes

The main script supports three modes via the `--mode` argument:

- `TRAIN`: Train the model.
- `TEST`: Test the model.
- `INFERENCE`: Perform inference (default).

When using `INFERENCE`, you can specify the index of the image to predict via the `--idx` argument.

Examples:

- Train the model:

  ```bash
  python main.py --mode TRAIN
  ```

- Test the model:

  ```bash
  python main.py --mode TEST
  ```

- Inference on image with index 5:

  ```bash
  python main.py --mode INFERENCE --idx 5
  ```

### Option 2. Run the interactive app with Streamlit

Start the Streamlit web interface with:

```bash
streamlit run Home.py
```

Open the browser at the URL provided to interact with the application.
