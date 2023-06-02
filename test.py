import numpy as np
import torch
import os
from tqdm import tqdm
from dataloader_onelabel import CustomDataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import matplotlib.pyplot as plt

with open('rs19_val\jpgs\\rs19_val\\rs00005.jpg', "rb") as image_file:
    image = Image.open(image_file)
    
    image = image.resize((224, 224), Image.BILINEAR)
    image = torch.from_numpy(np.array(image) / 255).view(3,224,224).float()
    image = image.unsqueeze(0)

    model = torch.load('models\\model_20_0.1') # model_20_0.01-ok model_80_0.01 model_30_0.01-top model_10_0.1? model_40_0.01
    model, image = model.cpu(), image.cpu()
    model.eval()
    output = model(image)
    output = output['out'].detach().numpy()

    #plt.imshow(mask_norm, cmap='gray')
    #plt.show()

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