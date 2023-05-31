import numpy as np
import torch
import os
from tqdm import tqdm
from dataloader_onelabel import CustomDataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import matplotlib.pyplot as plt

with open('rs19_val\jpgs\\rs19_val\\rs00091.jpg', "rb") as image_file, open('rs19_val\\uint8\\objects\\91_5.png', "rb") as mask_file:
    image = Image.open(image_file)
    mask = Image.open(mask_file)
    mask = mask.convert("L")
                
    image = image.resize((224, 224), Image.BILINEAR)
    mask = mask.resize((224, 224), Image.BILINEAR)

    image = pil_to_tensor(image).float()
    mask = pil_to_tensor(mask).float()
    mask_norm = torch.squeeze(mask / 254).long()

    image = image.unsqueeze(0)

    model = torch.load('models\\model_30_0.001_adam_1out')
    model, image = model.cpu(), image.cpu()
    model.eval()
    output = model(image)
    output = output['out'].detach().numpy()

for o in range(21):
    plt.imshow(output[0][o], cmap='gray')
    plt.show()


#dataset = CustomDataset(subset = 'Test')
#dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
#outputs_test = []
#i = 1
#for inputs_test, masks in tqdm(dataloader):
#    outputs_test = model(inputs_test)
#    outputs_test = outputs_test['out'].detach().numpy()
#    np.save(os.path.join('models','output_{}'.format(i)), outputs_test)
#    i += 1