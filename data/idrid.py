import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IDRIDDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_id = self.labels.iloc[idx, 0]
        label = int(self.labels.iloc[idx, 1])

        # Ensure .png extension
        if not image_id.endswith(".jpg"):
            image_id += ".jpg"

        img_path = os.path.join(self.image_dir, image_id)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Could not open image: {img_path}. Error: {e}")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
