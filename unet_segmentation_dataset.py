from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch

class UnetSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size

        self.images = sorted([p for p in self.image_dir.glob("*.png")])
        self.masks = sorted([p for p in self.mask_dir.glob("*.png")])

        assert len(self.images) == len(self.masks), "❌ Images and masks does not match."


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert("RGB")
        # Correct PIL resize: width first, height second
        image = image.resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)
        image = T.ToTensor()(image)  # convert to tensor after resize

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size[1], self.img_size[0]), resample=Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask).long()

        return image, mask
