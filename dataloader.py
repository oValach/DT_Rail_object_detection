from pathlib import Path
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from PIL import Image
import numpy as np

class CustomDataset(VisionDataset):
    def __init__(self, image_folder, mask_folder, seed, subset, test_val_fraction):
        self.image_folder = Path(image_folder)
        self.mask_folder = Path(mask_folder)
        self.test_val_fraction = test_val_fraction

        self.image_list = np.array(sorted(Path(self.image_folder).glob("*")))
        self.mask_list = np.array(sorted(Path(self.mask_folder).glob("*")))

        if seed:
            np.random.seed(seed)
            indices = np.arange(len(self.image_list))
            np.random.shuffle(indices)
            self.image_list = self.image_list[indices]
            self.mask_list = self.mask_list[indices]
        if subset == 'Train':
            self.image_names = self.image_list[:int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction*2)))]
            self.mask_names = self.mask_list[:int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction*2)))]
        elif subset == 'Test':
            self.image_names = self.image_list[int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction*2))):int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction)))]
            self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction*2))):int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction)))]
        elif subset == 'Val':
            self.image_names = self.image_list[int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction))):]
            self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction))):]
        else:
            print('Invalid data subset.')

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        mask_path = self.mask_names[idx] 

        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            image = Image.open(image_file)
            mask = Image.open(mask_file)
            mask = mask.convert("L")

            image = pil_to_tensor(image).float()
            mask = pil_to_tensor(mask).float()

            sample = [image, mask]
            return sample
