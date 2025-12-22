import torch
import os
import numpy as np
import cv2
import skimage.measure
import data.Dataloader as Dataloader
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from SDNet_Model import SDNet_net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = SDNet_net()
task = 'MedicalTest'
# model.load_state_dict(torch.load(r'.\save\FedAvgSDNet_global.pth'))
model.load_state_dict(torch.load(r'.\save\FedAvgSDNet_local_Medical_4.pth'))
IVTestDataset = Dataloader.getDataset(task)
testloader = DataLoader(IVTestDataset, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
num_para = count_parameters(model)
print(f"Number of parameters encoder: {num_para}")
def unnormalize(tensor):
    """反归一化一个张量"""
    for t in tensor:
        t.mul(0.5).add(0.5)
    return tensor

def savenp(tensor, path):
    y = tensor
    h = y.shape[2]
    w = y.shape[3]
    y = y * 255.0
    img_copy = y.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    img_copy = np.clip(img_copy, 0, 255)
    img_copy = img_copy.astype(np.uint8)[0, :, :, :]
    cv2.imwrite(path, img_copy)
    sd = np.std(img_copy)
    en = skimage.measure.shannon_entropy(img_copy)
    return sd, en
if task == 'IVTest':
    for i, (IVDataset) in enumerate(testloader):
        visible_img = IVDataset[0]
        infrared_img = IVDataset[1]
        visible_img, infrared_img = visible_img.to(device), infrared_img.to(device)
        print(visible_img.shape)
        result, _, _ = model(infrared_img, visible_img)
        save_dir = r"E:\Datasets\FedFusion\test\result/"
        sd, en = savenp((result) * 0.5 + 0.5, save_dir + str(i + 1) + ".png")
        print(sd, en)
elif task == 'MedicalTest':
    for i, (pet, mri) in enumerate(testloader):
        pet, mri = pet.to(device), mri.to(device)
        result, output1, output2 = model(mri, pet)
        save_dir = r"E:\Datasets\FedFusion\test\result/"
        sd, en = savenp((result) * 0.5 + 0.5, save_dir + str(i + 1) + ".png")
        print(sd, en)
elif task == 'METest':
    for i, (over, under, _) in enumerate(testloader):
        under, over = under.to(device), over.to(device)
        result, output1, output2 = model(over, under)
        save_dir = r"E:\Datasets\FedFusion\test\result/"
        sd, en = savenp((result) * 0.5 + 0.5, save_dir + str(i + 1) + ".png")
        print(sd, en)
elif task == 'MFTest':
    for i, (far, near) in enumerate(testloader):
        far, near = far.to(device), near.to(device)
        result, output1, output2 = model(far, near)
        save_dir = r"E:\Datasets\FedFusion\test\result/"
        sd, en = savenp((result) * 0.5 + 0.5, save_dir + str(i + 1) + ".png")
        print(sd, en)

