import os
import torch
import re
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from src.options import args_parser

class CustomIVDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.visible_images = sorted(os.listdir(os.path.join(main_dir, "cut_visible")))
        self.infrared_images = sorted(os.listdir(os.path.join(main_dir, "cut_infrared")))

        # self.pet_images = sorted(os.listdir(os.path.join(main_dir, "pet")))
        # self.mri_images = sorted(os.listdir(os.path.join(main_dir, "mri")))

        # 确保两个文件夹中图像数量相同
        assert len(self.visible_images) == len(self.infrared_images), \
            "The number of visible and infrared images should be the same"

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def __len__(self):
        return len(self.visible_images)

    def __getitem__(self, idx):
        visible_img_path = os.path.join(self.main_dir, "cut_visible", self.visible_images[idx])
        infrared_img_path = os.path.join(self.main_dir, "cut_infrared", self.infrared_images[idx])

        visible_img = read_image(visible_img_path)
        infrared_img = read_image(infrared_img_path)

        if self.transform:
            visible_img = self.transform(visible_img)
            infrared_img = self.transform(infrared_img)

        return visible_img, infrared_img

class CustomIVTestDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.visible_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "visible")))
        self.infrared_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "infrared")))

        # self.pet_images = sorted(os.listdir(os.path.join(main_dir, "pet")))
        # self.mri_images = sorted(os.listdir(os.path.join(main_dir, "mri")))

        # 确保两个文件夹中图像数量相同
        assert len(self.visible_images) == len(self.infrared_images), \
            "The number of visible and infrared images should be the same"

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            # transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

        self.single_channel_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            # transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])
    def sorted_alphanumeric(self, data):
        """
        对列表进行排序，使数字部分按数值排序而不是字典序
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    def __len__(self):
        return len(self.visible_images)

    def __getitem__(self, idx):
        visible_img_path = os.path.join(self.main_dir, "visible", self.visible_images[idx])
        infrared_img_path = os.path.join(self.main_dir, "infrared", self.infrared_images[idx])

        visible_img = read_image(visible_img_path)
        infrared_img = read_image(infrared_img_path)

        if self.transform:
            visible_img = self.single_channel_transform(visible_img)
            infrared_img = self.single_channel_transform(infrared_img)

        return visible_img, infrared_img

class CustomMedicalDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.pet_images = sorted(os.listdir(os.path.join(main_dir, "PET")))
        self.mri_images = sorted(os.listdir(os.path.join(main_dir, "MRI")))

        # self.pet_images = sorted(os.listdir(os.path.join(main_dir, "pet")))
        # self.mri_images = sorted(os.listdir(os.path.join(main_dir, "mri")))

        # 确保两个文件夹中图像数量相同
        assert len(self.pet_images) == len(self.mri_images), \
            "The number of visible and infrared images should be the same"

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def __len__(self):
        return len(self.pet_images)

    def __getitem__(self, idx):
        pet_img_path = os.path.join(self.main_dir, "PET", self.pet_images[idx])
        mri_img_path = os.path.join(self.main_dir, "MRI", self.mri_images[idx])

        pet_img = read_image(pet_img_path)
        mri_img = read_image(mri_img_path)

        if self.transform:
            pet_img = self.transform(pet_img)
            mri_img = self.transform(mri_img)

        return pet_img, mri_img

class CustomMedicalTestDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.pet_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "PET")))
        self.mri_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "MRI")))

        # self.pet_images = sorted(os.listdir(os.path.join(main_dir, "pet")))
        # self.mri_images = sorted(os.listdir(os.path.join(main_dir, "mri")))

        # 确保两个文件夹中图像数量相同
        assert len(self.pet_images) == len(self.mri_images), \
            "The number of visible and infrared images should be the same"

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])
    def sorted_alphanumeric(self, data):
        """
        对列表进行排序，使数字部分按数值排序而不是字典序
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    def __len__(self):
        return len(self.pet_images)

    def __getitem__(self, idx):
        pet_img_path = os.path.join(self.main_dir, "PET", self.pet_images[idx])
        mri_img_path = os.path.join(self.main_dir, "MRI", self.mri_images[idx])

        pet_img = read_image(pet_img_path)
        mri_img = read_image(mri_img_path)

        if self.transform:
            pet_img = self.transform(pet_img)
            mri_img = self.transform(mri_img)

        return pet_img, mri_img

class CustomNaturalDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.natural_images = sorted(os.listdir(os.path.join(main_dir, "20000")))

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def __len__(self):
        return len(self.natural_images)

    def __getitem__(self, idx):
        natural_img_path = os.path.join(self.main_dir, "20000", self.natural_images[idx])
        natural_img = read_image(natural_img_path)
        # 检查通道数并在必要时转换图像
        if natural_img.shape[0] == 1:
            # 如果是单通道图像，则复制通道以形成三通道图像
            natural_img = natural_img.repeat(3, 1, 1)

        if self.transform:
            natural_img = self.transform(natural_img)

        return natural_img


class CustomDoubleNaturalDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.natural_images = sorted(os.listdir(os.path.join(main_dir, "20000")))
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.natural_images)

    def __getitem__(self, idx):
        # 获取当前图像
        natural_img_path = os.path.join(self.main_dir, "20000", self.natural_images[idx])
        natural_img = read_image(natural_img_path)

        # 获取下一张图像，如果是最后一张，则返回第一张
        next_idx = (idx + 1) % len(self.natural_images)
        next_img_path = os.path.join(self.main_dir, "20000", self.natural_images[next_idx])
        next_img = read_image(next_img_path)

        if natural_img.shape[0] == 1:
            natural_img = natural_img.repeat(3, 1, 1)

        if next_img.shape[0] == 1:
            next_img = next_img.repeat(3, 1, 1)

        if self.transform:
           natural_img = self.transform(natural_img)
           next_img = self.transform(next_img)

        return natural_img, next_img

class CustomValNaturalDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.natural_images = sorted(os.listdir(os.path.join(main_dir, "val_data")))
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((512, 512)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.natural_images)

    def __getitem__(self, idx):
        # 获取当前图像
        natural_img_path = os.path.join(self.main_dir, "val_data", self.natural_images[idx])
        natural_img = read_image(natural_img_path)

        # 获取下一张图像，如果是最后一张，则返回第一张
        next_idx = (idx + 1) % len(self.natural_images)
        next_img_path = os.path.join(self.main_dir, "val_data", self.natural_images[next_idx])
        next_img = read_image(next_img_path)

        if natural_img.shape[0] == 1:
            natural_img = natural_img.repeat(3, 1, 1)

        if next_img.shape[0] == 1:
            next_img = next_img.repeat(3, 1, 1)

        if self.transform:
           natural_img = self.transform(natural_img)
           next_img = self.transform(next_img)

        return natural_img, next_img

class CustomMFDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.far_images = sorted(os.listdir(os.path.join(main_dir, "cut_far")))
        self.near_images = sorted(os.listdir(os.path.join(main_dir, "cut_near")))

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def __len__(self):
        return len(self.far_images)

    def __getitem__(self, idx):
        far_img_path = os.path.join(self.main_dir, "cut_far", self.far_images[idx])
        near_img_path = os.path.join(self.main_dir, "cut_near", self.near_images[idx])

        far_img = read_image(far_img_path)
        near_img = read_image(near_img_path)

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        return far_img, near_img

class CustomMFTestDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.far_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "far")))
        self.near_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "near")))

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            # transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def sorted_alphanumeric(self, data):
        """
        对列表进行排序，使数字部分按数值排序而不是字典序
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    def __len__(self):
        return len(self.far_images)

    def __getitem__(self, idx):
        far_img_path = os.path.join(self.main_dir, "far", self.far_images[idx])
        near_img_path = os.path.join(self.main_dir, "near", self.near_images[idx])

        far_img = read_image(far_img_path)
        near_img = read_image(near_img_path)

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        return far_img, near_img

class CustomMFGTDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.far_images = sorted(os.listdir(os.path.join(main_dir, "MFI_WHU_far")))
        self.near_images = sorted(os.listdir(os.path.join(main_dir, "MFI_WHU_near")))
        self.gt_images = sorted(os.listdir(os.path.join(main_dir, "MFI_WHU_gt")))

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def __len__(self):
        return len(self.far_images)

    def __getitem__(self, idx):
        far_img_path = os.path.join(self.main_dir, "MFI_WHU_far", self.far_images[idx])
        near_img_path = os.path.join(self.main_dir, "MFI_WHU_near", self.near_images[idx])
        gt_img_path = os.path.join(self.main_dir, "MFI_WHU_gt", self.near_images[idx])

        far_img = read_image(far_img_path)
        near_img = read_image(near_img_path)
        gt_img = read_image(gt_img_path)

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)
            gt_img = self.transform(gt_img)

        return far_img, near_img, gt_img

class CustomMFGTTestDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.far_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "MFI_WHU_far")))
        self.near_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "MFI_WHU_near")))
        self.gt_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "MFI_WHU_gt")))

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            # transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def sorted_alphanumeric(self, data):
        """
        对列表进行排序，使数字部分按数值排序而不是字典序
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    def __len__(self):
        return len(self.far_images)

    def __getitem__(self, idx):
        far_img_path = os.path.join(self.main_dir, "MFI_WHU_far", self.far_images[idx])
        near_img_path = os.path.join(self.main_dir, "MFI_WHU_near", self.near_images[idx])
        gt_img_path = os.path.join(self.main_dir, "MFI_WHU_gt", self.near_images[idx])

        far_img = read_image(far_img_path)
        near_img = read_image(near_img_path)
        gt_img = read_image(gt_img_path)

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)
            gt_img = self.transform(gt_img)

        return far_img, near_img, gt_img

def unnormalize(tensor):
    """反归一化一个张量"""
    for t in tensor:
        t.mul(0.5).add(0.5)
    return tensor


class CustomMEDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.me_over_images = sorted(os.listdir(os.path.join(main_dir, "me_over")))
        self.me_under_images = sorted(os.listdir(os.path.join(main_dir, "me_under")))
        self.me_gt_images = sorted(os.listdir(os.path.join(main_dir, "me_gt")))

        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def __len__(self):
        return len(self.me_under_images)

    def __getitem__(self, idx):
        over_img_path = os.path.join(self.main_dir, "me_over", self.me_over_images[idx])
        under_img_path = os.path.join(self.main_dir, "me_under", self.me_under_images[idx])
        gt_img_path = os.path.join(self.main_dir, "me_gt", self.me_gt_images[idx])

        over_img = read_image(over_img_path)
        under_img = read_image(under_img_path)
        gt_img = read_image(gt_img_path)

        if self.transform:
            over_img = self.transform(over_img)
            under_img = self.transform(under_img)
            gt_img = self.transform(gt_img)

        return under_img, over_img, gt_img

class CustomMETestDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.me_over_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "me_over")))
        self.me_under_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "me_under")))
        self.me_gt_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "me_gt")))

        # self.pet_images = sorted(os.listdir(os.path.join(main_dir, "pet")))
        # self.mri_images = sorted(os.listdir(os.path.join(main_dir, "mri")))

        # 定义转换操作
        self.transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            # transforms.Resize((512, 512)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def sorted_alphanumeric(self, data):
        """
        对列表进行排序，使数字部分按数值排序而不是字典序
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    def __len__(self):
        return len(self.me_under_images)

    def __getitem__(self, idx):
        over_img_path = os.path.join(self.main_dir, "me_over", self.me_over_images[idx])
        under_img_path = os.path.join(self.main_dir, "me_under", self.me_under_images[idx])
        gt_img_path = os.path.join(self.main_dir, "me_gt", self.me_gt_images[idx])

        over_img = read_image(over_img_path)
        under_img = read_image(under_img_path)
        gt_img = read_image(gt_img_path)

        if self.transform:
            over_img = self.transform(over_img)
            under_img = self.transform(under_img)
            gt_img = self.transform(gt_img)

        return under_img, over_img, gt_img

class CustomMEHPTestDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.me_over_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "over")))
        self.me_under_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "under")))
        self.me_gt_images = self.sorted_alphanumeric(os.listdir(os.path.join(main_dir, "GT")))


        # 定义转换操作
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
            # transforms.Resize((512, 512)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 适应单通道
        ])

    def sorted_alphanumeric(self, data):
        """
        对列表进行排序，使数字部分按数值排序而不是字典序
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    def __len__(self):
        return len(self.me_under_images)

    def __getitem__(self, idx):
        over_img_path = os.path.join(self.main_dir, "over", self.me_over_images[idx])
        under_img_path = os.path.join(self.main_dir, "under", self.me_under_images[idx])
        gt_img_path = os.path.join(self.main_dir, "GT", self.me_gt_images[idx])

        over_img = read_image(over_img_path)
        under_img = read_image(under_img_path)
        gt_img = read_image(gt_img_path)

        if self.transform:
            over_img = self.transform(over_img)
            under_img = self.transform(under_img)
            gt_img = self.transform(gt_img)

        return under_img, over_img, gt_img

def getDataset(dataset):
    path = r"/media/data1/swj/FedFusion/train"
    test_path = r'/media/data1/swj/FedFusion/test'
    coco_path = r'/media/data1/swj/FedFusion/GeFuNet'
    if dataset == 'IV':
        return CustomIVDataset(main_dir=path)
    elif dataset == 'IVTest':
        return CustomIVTestDataset(main_dir=test_path)
    elif dataset == 'Medical':
        return CustomMedicalDataset(main_dir=path)
    elif dataset == 'natural':
        return CustomNaturalDataset(main_dir=coco_path)
    elif dataset == 'doubleNatural':
        return CustomDoubleNaturalDataset(main_dir=coco_path)
    elif dataset == 'valNatural':
        return CustomValNaturalDataset(main_dir=path)
    elif dataset == 'MedicalTest':
        return CustomMedicalTestDataset(main_dir=test_path)
    elif dataset == 'MF':
        return CustomMFDataset(main_dir=path)
    elif dataset == 'MFTest':
        return CustomMFTestDataset(main_dir=test_path)
    elif dataset == 'ME':
        return CustomMEDataset(main_dir=path)
    elif dataset == 'METest':
        return CustomMETestDataset(main_dir=test_path)
    elif dataset == 'MFGT':
        return CustomMFGTDataset(main_dir=path)
    elif dataset == 'MFGTTest':
        return CustomMFGTTestDataset(main_dir=test_path)
    elif dataset == 'MEHPTest':
        return CustomMEHPTestDataset(main_dir=test_path)

# 使用示例
# arg = args_parser()
# IVdataset = CustomIVDataset(main_dir=r"E:\Datasets\FedFusion\train")
# IVdataloader = DataLoader(IVdataset, batch_size=arg.batch_size, shuffle=True)
