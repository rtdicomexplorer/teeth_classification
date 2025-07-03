from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class UnetSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(512, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size

        self.images = sorted([p for p in self.image_dir.glob("*.png")])
        self.masks = sorted([p for p in self.mask_dir.glob("*.png")])

        assert len(self.images) == len(self.masks), "❌ Images and masks does not match."

        self.image_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor()
        ])

        self.mask_transform = T.Compose([
            T.Resize(self.img_size, interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # L = 1 channel

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # binary Masks (0 oder 1)

        return image, mask
