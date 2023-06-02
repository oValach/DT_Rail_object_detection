from pathlib import Path
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from PIL import Image
import numpy as np
import torch

class CustomDataset(VisionDataset):
    def __init__(self, image_folder, mask_folder, seed, subset, test_val_fraction = 0.1):
        self.image_folder = Path(image_folder)
        self.mask_folder = Path(mask_folder)
        self.test_val_fraction = test_val_fraction

        # all files
        self.image_list = np.array(sorted(Path(self.image_folder).glob("*")))
        self.mask_list = np.array(sorted(Path(self.mask_folder).glob("*")))

        for file_path in self.image_list:
            if 'desktop.ini' in file_path.name:
                file_path.unlink()
        for file_path in self.mask_list:
            if 'desktop.ini' in file_path.name:
                file_path.unlink()

        if seed: #rng locked data shuffle and split
            np.random.seed(seed)
            indices = np.arange(len(self.image_list))
            np.random.shuffle(indices)
            self.image_list = self.image_list[indices]
            self.mask_list = self.mask_list[indices]
        if subset == 'Train': # split dataset to 1-2*fraction of train data, default fraction == 0.1
            self.image_names = self.image_list[:int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction*2)))]
            self.mask_names = self.mask_list[:int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction*2)))]
        elif subset == 'Test': # test data of lenght of fraction
            self.image_names = self.image_list[int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction*2))):int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction)))]
            self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction*2))):int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction)))]
        elif subset == 'Val': # val data - different part of data than test, also of length fraction
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
            mask_data = np.load(mask_file)
            mask = mask_data['arr_0']

            image = image.resize((224, 224), Image.BILINEAR)
            mask = np.resize(mask, (224, 224))

            image = torch.from_numpy(np.array(image) / 255).view(3,224,224).float()
            
            mask = torch.from_numpy(mask).float()
            mask_norm = torch.squeeze(mask / 255).long()

            sample = [image, mask_norm]
            return sample