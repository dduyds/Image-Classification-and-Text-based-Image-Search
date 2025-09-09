from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms as T
from model import Tokenizer

class Cifar10(Dataset):
    def __init__(self, train=True):
        self.dataset = load_dataset("uoft-cs/cifar10")

        self.transform = T.ToTensor()

        if train:
            self.split = "train"
        else:
            self.split = "test"


        self.captions = {0: "An image of airplane",
                          1: "An image of automobile",
                          2: "An image of bird",
                          3: "An image of cat",
                          4: "An image of deer",
                          5: "An image of dog",
                          6: "An image of frog",
                          7: "An image of horse",
                          8: "An image of ship",
                          9: "An image of truck"}


    def __len__(self):
        return self.dataset.num_rows[self.split]

    def __getitem__(self, index):
        # Lấy ảnh
        img = self.dataset[self.split][index]["img"]
        img = self.transform(img)

        # Lấy caption và mask
        cap, mask = Tokenizer(self.captions[self.dataset[self.split][index]["label"]])

        # Rezise mặt mạ
        mask = mask.repeat(len(mask),1)

        return {"image": img, "caption": cap, "mask": mask}