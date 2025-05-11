import numpy as np
import timm
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from sklearn.manifold import TSNE
from torchvision import transforms

from testttt import MyMoudle
from .Swin import SwinTransformer
from .clip import clip
from PIL import Image, ImageFilter
import torch.nn as nn

CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta


class HighPassFilter:
    def __init__(self, cutoff_frequency):
        self.cutoff_frequency = cutoff_frequency

    def __call__(self, x):
        # 假设x是一个PIL Image
        filtered_image = x.filter(ImageFilter.ModeFilter(self.cutoff_frequency))
        return filtered_image


class SimpleUpsampleLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleUpsampleLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name,
                                                device="cpu")  # self.preprecess will not be used during training, which is handled in Dataset class

        self.fc = nn.Linear(CHANNELS[name], num_classes)
        #---------------------------------------------------
        self.MyMoudle = MyMoudle(768,16,no_spatial=True)

    def forward(self, x, return_feature=False):
        #----------------------------------------
        # print(x.shape)
        features = self.model.encode_image(x)

        #----------------------------------------
        # features = features.unsqueeze(2)
        # features = features.unsqueeze(3)
        # features = self.MyMoudle(features)
        # features = features.squeeze(3)
        # features = features.squeeze(2)
        # print(features.shape)
        #---------------------------------------------
        # mask = (torch.rand_like(features) > 0.9).float()  # 70% 保留，30% 丢失
        # masked_features = features * mask
        #
        # upsample_layer = SimpleUpsampleLayer(768, 224 * 224).to(device)  # 将特征映射到图像空间
        # upsampled_features = upsample_layer(masked_features)  # 形状为 [4, 224 * 224]
        # upsampled_features = upsampled_features.view(2, 1, 224, 224)
        # heatmaps = upsampled_features.squeeze().detach().cpu().numpy()  # 转换为NumPy数组

        # 2. 自定义色彩映射
        # 自定义色彩映射，从蓝色到红色
        # cmap = LinearSegmentedColormap.from_list(
        #     "cold_to_warm", ['cyan',  "yellow", "red"]
        # )
        #
        # # 3. 生成热力图并叠加在原始图像
        # for i in range(4):
        #     img = x[i].permute(1, 2, 0).detach().cpu().numpy()  # 转换为NumPy数组并调换维度
        #
        #     # 归一化热力图
        #     heatmap = heatmaps[i]
        #     heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # 归一化到[0, 1]
        #
        #     # 应用高斯模糊
        #     heatmap = gaussian_filter(heatmap, sigma=3)  # 可以调整sigma值以控制模糊程度
        #     # 叠加热力图到原始图像上
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(img)
        #     plt.imshow(heatmap, alpha=0.5, cmap=cmap)  # 将热力图叠加到原始图像上
        #     plt.colorbar()
        #     plt.title(f'Heatmap for image {i}')
        #     plt.axis('off')
        #     plt.show()
        #---------------------------------------------
        if return_feature:
            return features
        # print(self.fc(features))
        return self.fc(features)
