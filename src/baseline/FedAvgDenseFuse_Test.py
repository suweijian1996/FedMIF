import torch
import os
import numpy as np
import cv2
import skimage.measure
import data.Dataloader as Dataloader
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from DenseFuse_Model import DenseFuse_net

task = 'MedicalTest'
model = DenseFuse_net()
# model.load_state_dict(torch.load(r'.\save\FedAvgDenseFuse_global.pth'))
model.load_state_dict(torch.load(r'.\save\FedAvgDenseFuse_local_4.pth'))
IVTestDataset = Dataloader.getDataset(task)
testloader = DataLoader(IVTestDataset, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
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

if task == 'METest':
    for i, (over, under, _) in enumerate(testloader):
        under, over = under.to(device), over.to(device)
        vis_en = model.encoder(under)
        ir_en = model.encoder(over)
        fusion_en = model.fusion(vis_en, ir_en)
        result = model.decoder(fusion_en)
        save_dir = r"E:\Datasets\FedFusion\test\result/"
        sd, en = savenp((result) * 0.5 + 0.5, save_dir + str(i + 1) + ".png")
        print(sd, en)
else:
    for i, (IVDataset) in enumerate(testloader):
        visible_img = IVDataset[0]
        infrared_img = IVDataset[1]
        visible_img, infrared_img = visible_img.to(device), infrared_img.to(device)
        print(visible_img.shape)
        vis_en = model.encoder(visible_img)
        ir_en = model.encoder(infrared_img)
        fusion_en = model.fusion(vis_en, ir_en)
        result = model.decoder(fusion_en)
        # ir = ir/2 + 0.5
        # vis = vis/2 + 0.5

        # result = unnormalize(result)
        # result = result + visible_img
        # result = vis - result

        # result = unnormalize(visible_img - result)
        #
        #     # 保存目录
        save_dir = r"E:\Datasets\FedFusion\test\result/"
        # #     if not os.path.exists(save_dir):
        # #         os.makedirs(save_dir)
        # #
        # #     # 保存图像
        sd, en = savenp((result) * 0.5 + 0.5, save_dir + str(i + 1) + ".png")
        print(sd, en)
        # save_image(infrared_img, os.path.join(save_dir, f"fused_{i}.png"))