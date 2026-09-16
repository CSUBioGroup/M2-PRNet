import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import re

from tqdm import tqdm

class ImageProcessor:
    """图像预处理器，一次性加载所有图像到内存"""
    
    def __init__(self, image_dir, hidden_dim=256):
        self.image_dir = image_dir
        self.hidden_dim = hidden_dim
        self.transform = self._get_transform()
        self.images_cache = {}  # 缓存所有图像
        cache_file="all_images_cache.pt"
        self.cache_file = os.path.join(image_dir, cache_file)
        # self.load_all_images()
        if os.path.exists(self.cache_file):
            print(f"Loading cached image embeddings from {self.cache_file} ...")
            self.images_cache = torch.load(self.cache_file)
            # print(self.images_cache.keys())
            print(f"[OK] Loaded {len(self.images_cache)} cached image groups")
        else:
            # 没缓存 → 正常读取 PNG → 自动缓存
            self.load_all_images()
            print(f"Saving image cache to {self.cache_file}")
            torch.save(self.images_cache, self.cache_file)
        
    def _get_transform(self):
        """定义图像预处理变换"""
        return transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_pdbid_modelid(self, filename):
        """从文件名中提取pdbid和modelid"""
        # 匹配格式: pdbid_model{modelid}_front/side/top.png
        pattern = r'(.+)_model(\d+)_(front|side|top)\.png'
        match = re.match(pattern, filename)
        if match:
            pdbid = match.group(1)
            modelid = match.group(2)
            view_type = match.group(3)
            return pdbid, modelid, view_type
        return None, None, None
    
    def load_all_images(self):
        """一次性加载所有图像到内存"""
        print("Loading all images into memory...")
        image_files = glob.glob(os.path.join(self.image_dir, "*.png"))
        image_files = [img for img in image_files if 'small.png' not in img]
        for img_path in tqdm(image_files):
            filename = os.path.basename(img_path)
            pdbid, modelid, view_type = self.extract_pdbid_modelid(filename)
            # #case
            # if self.args.data_set == 'data/case/case.csv':
            #     pdbid = filename.split('_')[0] if '_' in filename else filename[0:4]
            #     modelid = filename.split('_')[1] 
            #     view_type = filename.split('_')[2].split('.')[0]
            # elif self.args.data_set == 'data/vegf/vegf_case.csv':
            pdbid = filename.split('_model')[0] if '_model' in filename else filename[0:4]
            # modelid = filename.split('_model_')[1][0] 
            modelid = filename.split('_')[4].split('.')[0]
            view_type = filename.split('_')[5].split('.')[0]
            # pdbid = filename.split('_')[0] if '_' in filename else filename[0:4]
            # modelid = '0'
            view_type = filename.split('_')[1].split('.')[0]
            fs = filename.split('_')
            # if pdbid and modelid:
            if 'mut' in filename:
                key = f"{fs[0]}_mut_{fs[2]}.pdb"  # 与图网络数据对齐的key
            else:
                key = f"{pdbid}_model_{modelid}.pdb"
            print(key)
            # key = filename
            if key not in self.images_cache:
                self.images_cache[key] = {}
            
            try:
                image = Image.open(img_path).convert('RGB')
                print(img_path)
                image_tensor = self.transform(image)
                self.images_cache[key][view_type] = image_tensor
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        print(f"Loaded {len(self.images_cache)} protein images with {sum(len(views) for views in self.images_cache.values())} total views")
    
    def get_image_views(self, key):
        """根据key获取三视图图像"""
        key = key + ".pdb"  # 确保key格式一致
        #case
        # key = key
        # print(self.images_cache.keys())
        # if self.args.data_set == 'data/case/case.csv':
        # key = key.split('_')[0]+'_' + key.split('_')[2][0]+'.pdb'  #case
        if key in self.images_cache:
            views = self.images_cache[key]
            # print(f"Found images for key: {key} with views: {list(views.keys())}")
            # 确保三个视图都存在，如果缺少某个视图用零张量替代
            front = views.get('front', torch.zeros(3, 224, 224))
            side = views.get('side', torch.zeros(3, 224, 224))
            top = views.get('top', torch.zeros(3, 224, 224))
            return front, side, top
        else:
            print('not found image for key:', key)
            # 如果没有对应的图像，返回零张量
            zero_tensor = torch.zeros(3, 224, 224)
            return zero_tensor, zero_tensor, zero_tensor

# class ImageNetwork(nn.Module):
#     """图像特征提取网络"""
    
#     def __init__(self, hidden_dim=256, pretrained=True):
#         super(ImageNetwork, self).__init__()
#         self.hidden_dim = hidden_dim
        
#         # 使用预训练的ResNet作为基础网络
#         resnet = models.resnet50(pretrained=pretrained)
        
#         # 移除最后的全连接层
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
#         # 获取ResNet的特征维度
#         resnet_feature_dim = 2048  # ResNet50最后一层的特征维度
        
#         # 为每个视图单独的特征变换
#         self.view_projectors = nn.ModuleDict({
#             'front': nn.Linear(resnet_feature_dim, hidden_dim),
#             'side': nn.Linear(resnet_feature_dim, hidden_dim),
#             'top': nn.Linear(resnet_feature_dim, hidden_dim)
#         })
        
#         # 视图融合网络
#         self.fusion_net = nn.Sequential(
#             nn.Linear(hidden_dim * 3, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
        
#         # 初始化权重
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         """初始化权重"""
#         for view in ['front', 'side', 'top']:
#             nn.init.xavier_uniform_(self.view_projectors[view].weight)
#             nn.init.zeros_(self.view_projectors[view].bias)
        
#         for layer in self.fusion_net:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.bias)
    
#     def forward(self, front_images, side_images, top_images):
#         """
#         前向传播
#         Args:
#             front_images: [batch_size, 3, 224, 224]
#             side_images: [batch_size, 3, 224, 224]
#             top_images: [batch_size, 3, 224, 224]
#         Returns:
#             image_features: [batch_size, hidden_dim]
#         """
#         batch_size = front_images.size(0)
        
#         # 提取各视图特征
#         view_features = {}
#         for view_name, images in zip(['front', 'side', 'top'], 
#                                    [front_images, side_images, top_images]):
#             # 通过ResNet提取特征 [batch_size, 2048, 1, 1]
#             features = self.feature_extractor(images)
#             # 展平 [batch_size, 2048]
#             features = features.view(batch_size, -1)
#             # 投影到目标维度 [batch_size, hidden_dim]
#             view_features[view_name] = self.view_projectors[view_name](features)
        
#         # 拼接三个视图的特征 [batch_size, hidden_dim * 3]
#         concatenated_features = torch.cat([
#             view_features['front'],
#             view_features['side'], 
#             view_features['top']
#         ], dim=1)
        
#         # 融合特征 [batch_size, hidden_dim]
#         fused_features = self.fusion_net(concatenated_features)
        
#         return fused_features

# class MultiViewImageNetwork(nn.Module):
#     """多视图图像网络的简化版本，直接输出256维特征"""
    
#     def __init__(self, hidden_dim=256):
#         super(MultiViewImageNetwork, self).__init__()
#         self.hidden_dim = hidden_dim
        
#         # 使用更轻量的ResNet18
#         resnet = models.resnet18(pretrained=True)
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
#         resnet_feature_dim = 512  # ResNet18的特征维度
        
#         # 直接映射到目标维度
#         self.feature_projector = nn.Sequential(
#             nn.Linear(resnet_feature_dim * 3, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
    
#     def forward(self, front_images, side_images, top_images):
#         batch_size = front_images.size(0)
        
#         # 提取三个视图的特征
#         front_feat = self.feature_extractor(front_images).view(batch_size, -1)
#         side_feat = self.feature_extractor(side_images).view(batch_size, -1)
#         top_feat = self.feature_extractor(top_images).view(batch_size, -1)
        
#         # 拼接并投影
#         combined = torch.cat([front_feat, side_feat, top_feat], dim=1)
#         return self.feature_projector(combined)