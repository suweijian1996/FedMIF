import torch
import os
import sys

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import numpy as np
import cv2
import skimage.measure
import data.Dataloader as Dataloader
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from FedFusion_model import FedMIFModel, FedMIFModel_GN
from options import args_parser

args = args_parser()

if args.norm == 'None': 
    model = FedMIFModel()
    main_path = 'FedMIF'
elif args.norm == 'GN':
    model = FedMIFModel_GN()
    main_path = 'FedMIF_GN'

model.load_state_dict(torch.load(f'./pth/{main_path}/global.pth'))
model.fusion_decoder.load_state_dict(torch.load(f'./pth/{main_path}/global_IV_decoder.pth'))
model.detail_net.load_state_dict(torch.load(f'./pth/{main_path}/global_detail_IV_0.pth'))

IVTestDataset = Dataloader.getDataset('IVTest')
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

for i, (IVDataset) in enumerate(testloader):
    visible_img = IVDataset[0]
    infrared_img = IVDataset[1]
    visible_img, infrared_img = visible_img.to(device), infrared_img.to(device)
    print(visible_img.shape)
    global_result = model.feature_fusion(torch.cat((infrared_img, visible_img), 1))
    adapter_result = model.detail_net(torch.cat((infrared_img, visible_img), 1))
    result = global_result + adapter_result

#     # 保存目录
    save_dir = r"./results/"
    global_dir = save_dir + "global/"
    client_dir = save_dir + "client/"
    if not os.path.exists(global_dir):
        os.makedirs(global_dir)
        
    if not os.path.exists(client_dir):
        os.makedirs(client_dir)

# #     # 保存图像
    sd, en = savenp((result) * 0.5 + 0.5, client_dir" + str(i+1) + ".png")
    sd, en = savenp((global_result) * 0.5 + 0.5, global_dir + str(i+1) + ".png")
    print(sd, en)

    # _, _ = savenp((adapter_result) * 0.5 + 0.5, save_dir + str(i+1) + "_adapter_" + ".png")
     # _, _ = savenp((global_result) * 0.5 + 0.5, save_dir + str(i+1) + "_global_" + ".png")
