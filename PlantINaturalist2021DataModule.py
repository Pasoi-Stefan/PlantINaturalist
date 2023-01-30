import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


class PlantINaturalist2021DataModule(pl.LightningDataModule):
    def __init__(self, transform, batch_size = 32, num_workers = 0, pin_memory = False , data_dir = "./"):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_dir = data_dir


    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.inaturalist_train = ImageFolder(f"{self.data_dir}/PlantINaturalist2021(2021_train_mini)_250/train", transform=self.transform)
            self.inaturalist_valid = ImageFolder(f"{self.data_dir}/PlantINaturalist2021(2021_train_mini)_250/validation", transform=TRANSFORM)

    def train_dataloader(self):
        return DataLoader(self.inaturalist_train, batch_size = self.batch_size, num_workers = self.num_workers, pin_memory = self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.inaturalist_valid, batch_size = self.batch_size, num_workers = self.num_workers, pin_memory = self.pin_memory)