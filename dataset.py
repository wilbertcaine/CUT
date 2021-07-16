from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class XYDataset(Dataset):
    def __init__(self, root_X, root_Y, transform=None):
        self.root_X = root_X
        self.root_Y = root_Y
        self.transform = transform

        self.X_images = os.listdir(root_X)
        self.Y_images = os.listdir(root_Y)
        self.length_dataset = max(len(self.X_images), len(self.Y_images))
        self.X_len = len(self.X_images)
        self.Y_len = len(self.Y_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        X_img = self.X_images[index % self.X_len]
        Y_img = self.Y_images[index % self.Y_len]

        X_path = os.path.join(self.root_X, X_img)
        Y_path = os.path.join(self.root_Y, Y_img)

        X_img = np.array(Image.open(X_path).convert("RGB"))
        Y_img = np.array(Image.open(Y_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=X_img, image0=Y_img)
            X_img = augmentations["image"]
            Y_img = augmentations["image0"]

        return X_img, Y_img
