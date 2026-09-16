import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import timm
import torch.nn.functional as F
class TrainableImageNetwork(nn.Module):
    """可训练的图像特征提取网络"""
    
    def __init__(self, hidden_dim=256, pretrained=True):
        super(TrainableImageNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 使用预训练的ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # 解冻最后几层进行训练
        for param in resnet.parameters():
            param.requires_grad = False
        
        # 解冻最后两个block进行微调
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        for param in resnet.layer4.parameters():
            param.requires_grad = True
            
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        resnet_feature_dim = 512
        
        # 特征投影层
        self.feature_projector = nn.Sequential(
            nn.Linear(resnet_feature_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.feature_projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, front_images, side_images, top_images):
        batch_size = front_images.size(0)
        
        # 提取三个视图的特征
        front_feat = self.feature_extractor(front_images).view(batch_size, -1)
        side_feat = self.feature_extractor(side_images).view(batch_size, -1)
        top_feat = self.feature_extractor(top_images).view(batch_size, -1)
        
        # 拼接并投影
        combined = torch.cat([front_feat, side_feat, top_feat], dim=1)
        return self.feature_projector(combined)

# class MultiViewConvNeXt(nn.Module):
#     def __init__(self, args, model_name='convnext_base'):
#         super().__init__()
#         self.args = args
        
#         # 创建三个独立的ConvNeXt backbone（权重共享或不共享）
#         self.share_weights = args.share_weights  # 是否共享三视图的权重
        
#         if self.share_weights:
#             # 权重共享：三个视图使用同一个backbone
#             self.backbone = timm.create_model(
#                 model_name, 
#                 pretrained=True, 
#                 num_classes=0,  # 移除分类头
#                 features_only=False,
#                 # out_indices=( 3)  # 指定要输出的层索引
#             )
#             self.axial_backbone = self.backbone
#             self.coronal_backbone = self.backbone  
#             self.sagittal_backbone = self.backbone
#         else:
#             # 权重不共享：每个视图使用独立的backbone
#             self.axial_backbone = timm.create_model(
#                 model_name, 
#                 pretrained=True, 
#                 num_classes=0,  # 移除分类头
#                 features_only=True,
#                 out_indices=( 3)  # 指定要输出的层索引
#             )
#             self.coronal_backbone = timm.create_model(
#                 model_name, 
#                 pretrained=True, 
#                 num_classes=0,  # 移除分类头
#                 features_only=True,
#                 out_indices=(3)  # 指定要输出的层索引
#             )
#             self.sagittal_backbone = timm.create_model(
#                 model_name, 
#                 pretrained=True, 
#                 num_classes=0,  # 移除分类头
#                 features_only=True,
#                 out_indices=(3)  # 指定要输出的层索引
#             )
#         # 获取各层的输出通道数
#         # self.feature_channels = self.backbone.feature_info.channels()
#         # 获取特征维度
#         self.feature_dim = self.axial_backbone.num_features
#         # 正确获取特征通道数
#         # self.feature_info = self.axial_backbone.feature_info
#         # self.feature_channels = self.feature_info.channels()
#         # self.feature_reductions = self.feature_info.reduction()
        
#         # print(f"各层输出通道数: {self.feature_channels}")
#         # print(f"各层下采样率: {self.feature_reductions}")
        
#         # 计算总特征维度
#         # self.total_features = sum(self.feature_channels)
#         # self.feature_dim = self.total_features  # 或者你可以选择其他投影维度
#         # 视图特征融合模块
#         self.view_fusion = ViewFusionModule(self.feature_dim)
        
#         # 投影层（如果需要调整维度）
#         self.feature_proj = nn.Linear(self.feature_dim, args.n_out_feature)
#         self._freeze_backbone()
#         # self._initialize_weights()
#     # def _initialize_weights(self):
#         # nn.init.xavier_uniform_(self.feature_proj.weight)
#         # nn.init.zeros_(self.feature_proj.bias)
#         # for layer in self.feature_proj:
#         #     if isinstance(layer, nn.Linear):
#         #         nn.init.xavier_uniform_(layer.weight)
#         #         nn.init.zeros_(layer.bias)
#     def _freeze_backbone(self):
#         """冻结backbone参数，但最后两层不冻结"""
#         # 获取所有层名
#         layer_names = []
#         for name, _ in self.axial_backbone.named_parameters():
#             # 提取层标识（如 'layer1.0.conv1' -> 'layer1'）
#             layer_id = name.split('.')[0]
#             if layer_id not in layer_names:
#                 layer_names.append(layer_id)
#         print(layer_names)
#         # 取最后两层不冻结
#         if len(layer_names) >= 2:
#             unfreeze_layers = layer_names[-1:]
#         else:
#             unfreeze_layers = layer_names
        
#         # 冻结参数
#         for name, param in self.axial_backbone.named_parameters():
#             layer_id = name.split('.')[0]
#             # if layer_id not in unfreeze_layers:
#             param.requires_grad = False
        
#         print(f"✓ Backbone参数已冻结，仅 {unfreeze_layers} 层可训练")
#     def _extract_multi_level_features(self, x, mode='axial'):
#         """提取多层特征并融合"""
#         # 获取所有指定层的特征 [batch, channels, H, W]
#         if self.share_weights:
#             features = self.backbone(x)[0]  # 返回包含4个特征的列表
#         else:
#             if mode == 'axial':
#                 features = self.axial_backbone(x)[0]  # 返回包含4个特征的列表
#             elif mode == 'coronal':
#                 features = self.coronal_backbone(x)[0]  # 返回包含4个特征的列表    
#             elif mode == 'sagittal':
#                 features = self.sagittal_backbone(x)[0] # 返回包含4个特征的列表
#         # print("axial",self.axial_backbone(x).shape )
#         # features = self.backbone(x)  # 返回包含4个特征的列表
#         return features
#         # 对每个特征图进行全局平均池化
#         pooled_features = []
#         for feat in features:
#             pooled = F.adaptive_avg_pool2d(feat, (1, 1))  # [batch, C, 1, 1]
#             pooled = pooled.view(pooled.size(0), -1)      # [batch, C]
#             pooled_features.append(pooled)
        
#         # 拼接所有层的特征
#         concatenated = torch.cat(pooled_features, dim=1)  # [batch, sum(C)]
        
#         return concatenated
#     def forward(self, axial_view, coronal_view, sagittal_view):
#         """
#         axial_view: 轴向视图 [batch, channels, height, width]
#         coronal_view: 冠状视图 [batch, channels, height, width]  
#         sagittal_view: 矢状视图 [batch, channels, height, width]
#         """
#         batch_size = axial_view.shape[0]
        
#         # 提取各视图特征
#         axial_features = self.axial_backbone(axial_view)  # [batch, feature_dim]
#         coronal_features = self.coronal_backbone(coronal_view)
#         sagittal_features = self.sagittal_backbone(sagittal_view)
#         # axial_features = self._extract_multi_level_features(axial_view,mode = 'axial')
#         # coronal_features = self._extract_multi_level_features(coronal_view ,mode = 'coronal')
#         # sagittal_features = self._extract_multi_level_features(sagittal_view ,mode = 'sagittal')
        
#         # 视图特征融合
#         fused_view_features = self.view_fusion(
#             axial_features, coronal_features, sagittal_features
#         )
        
#         # 投影到目标维度
#         projected_features = self.feature_proj(fused_view_features)
        
#         return projected_features
# class ViewFusionModule(nn.Module):
#     """三视图特征融合模块"""
#     def __init__(self, feature_dim):
#         super().__init__()
#         self.feature_dim = feature_dim
        
#         # 注意力权重学习
#         self.view_attention = nn.MultiheadAttention(
#             embed_dim=feature_dim, num_heads=8, batch_first=True
#         )
        
#         # 门控融合
#         self.gate_axial = nn.Linear(feature_dim, feature_dim)
#         self.gate_coronal = nn.Linear(feature_dim, feature_dim)
#         self.gate_sagittal = nn.Linear(feature_dim, feature_dim)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, axial_feat, coronal_feat, sagittal_feat):
#         # 将三个视图特征堆叠 [batch, 3, feature_dim]
#         view_features = torch.stack([axial_feat, coronal_feat, sagittal_feat], dim=1)
        
#         # 使用注意力机制融合视图特征
#         attended_features, _ = self.view_attention(
#             view_features, view_features, view_features
#         )
        
#         # 门控加权融合
#         gate_weights = self.sigmoid(
#             self.gate_axial(axial_feat) + 
#             self.gate_coronal(coronal_feat) + 
#             self.gate_sagittal(sagittal_feat)
#         )
        
#         # 平均池化 + 门控
#         mean_features = attended_features.mean(dim=1)  # [batch, feature_dim]
#         fused_features = gate_weights * mean_features
        
#         return fused_features

class MultiViewConvNeXt(nn.Module):
    def __init__(self, args, model_name='convnext_base'):
        super().__init__()
        self.args = args
        
        # 创建三个独立的ConvNeXt backbone（权重共享或不共享）
        self.share_weights = args.share_weights  # 是否共享三视图的权重
        
        if self.share_weights:
            # 权重共享：三个视图使用同一个backbone
            self.backbone = timm.create_model(
                model_name, 
                pretrained=True, 
                num_classes=0,  # 移除分类头
                features_only=True,  # 启用多层特征提取
                out_indices=(0, 1, 2, 3)  # 指定要输出的四层索引
            )
            self.axial_backbone = self.backbone
            self.coronal_backbone = self.backbone  
            self.sagittal_backbone = self.backbone
        else:
            # 权重不共享：每个视图使用独立的backbone
            self.axial_backbone = timm.create_model(
                model_name, 
                pretrained=True, 
                num_classes=0,  # 移除分类头
                features_only=True,  # 启用多层特征提取
                out_indices=(0, 1, 2, 3)  # 指定要输出的四层索引
            )
            self.coronal_backbone = timm.create_model(
                model_name, 
                pretrained=True, 
                num_classes=0,  # 移除分类头
                features_only=True,  # 启用多层特征提取
                out_indices=(0, 1, 2, 3)  # 指定要输出的四层索引
            )
            self.sagittal_backbone = timm.create_model(
                model_name, 
                pretrained=True, 
                num_classes=0,  # 移除分类头
                features_only=True,  # 启用多层特征提取
                out_indices=(0, 1, 2, 3)  # 指定要输出的四层索引
            )
        
        # 获取各层的输出通道数
        if self.share_weights:
            feature_info = self.backbone.feature_info
        else:
            feature_info = self.axial_backbone.feature_info
            
        self.feature_channels = feature_info.channels()
        self.feature_reductions = feature_info.reduction()
        
        print(f"各层输出通道数: {self.feature_channels}")
        print(f"各层下采样率: {self.feature_reductions}")
        
        # 计算总特征维度
        self.total_features = sum(self.feature_channels)
        
        # 各层特征投影层（将不同层特征投影到统一维度）
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 64),  # 将每层特征投影到64维
                nn.ReLU(),
                nn.Dropout(0.4)
            ) for channels in self.feature_channels
        ])
        
        # 每个视图的特征维度：4层 * 64维 = 256维
        self.view_feature_dim = 64 * len(self.feature_channels)
        
        # 视图特征融合模块
        self.view_fusion = ViewFusionModule(self.view_feature_dim)
        
        # 投影层（如果需要调整维度）
        self.feature_proj = nn.Linear(self.view_feature_dim, args.n_out_feature)
        
        self._freeze_backbone()
        # self._initialize_weights()

    # def _freeze_backbone(self):
    #     """冻结backbone参数，但最后两层不冻结"""
    #     if self.share_weights:
    #         backbone = self.backbone
    #     else:
    #         backbone = self.axial_backbone
            
    #     # 获取所有层名
    #     layer_names = []
    #     for name, _ in backbone.named_parameters():
    #         # 提取层标识
    #         parts = name.split('.')
    #         if len(parts) >= 2 and parts[0] == 'stages':
    #             layer_id = f"stages.{parts[1]}"
    #             if layer_id not in layer_names:
    #                 layer_names.append(layer_id)
        
    #     # 取最后两层不冻结
    #     if len(layer_names) >= 2:
    #         unfreeze_layers = layer_names[-2:]
    #     else:
    #         unfreeze_layers = layer_names
        
    #     # 冻结参数
    #     backbones = [self.axial_backbone, self.coronal_backbone, self.sagittal_backbone] if not self.share_weights else [self.backbone]
        
    #     for backbone in backbones:
    #         for name, param in backbone.named_parameters():
    #             parts = name.split('.')
    #             if len(parts) >= 2 and parts[0] == 'stages':
    #                 layer_id = f"stages.{parts[1]}"
    #                 if layer_id not in unfreeze_layers:
    #                     param.requires_grad = False
        
    #     print(f"✓ Backbone参数已冻结，仅 {unfreeze_layers} 层可训练")
    def _freeze_backbone(self):
        """冻结backbone参数，但最后两层不冻结"""
        # 获取所有层名
        layer_names = []
        for name, _ in self.axial_backbone.named_parameters():
            # 提取层标识（如 'layer1.0.conv1' -> 'layer1'）
            layer_id = name.split('.')[0]
            if layer_id not in layer_names:
                layer_names.append(layer_id)
        print(layer_names)
        # 取最后四层不冻结
        if len(layer_names) >= 4:
            unfreeze_layers = layer_names[-4:-4]
        else:
            unfreeze_layers = layer_names
        
        # 冻结参数
        for name, param in self.axial_backbone.named_parameters():
            layer_id = name.split('.')[0]
            # if layer_id not in unfreeze_layers:
            param.requires_grad = False
    def _extract_multi_level_features(self, x, mode='axial'):
        """提取多层特征并融合"""
        # 获取四层特征 [batch, channels, H, W]
        if self.share_weights:
            features = self.backbone(x)  # 返回包含4个特征的列表
        else:
            if mode == 'axial':
                features = self.axial_backbone(x)
            elif mode == 'coronal':
                features = self.coronal_backbone(x)  
            elif mode == 'sagittal':
                features = self.sagittal_backbone(x)
        
        # 对每层特征进行投影和融合
        fused_features = []
        for i, (feat, proj) in enumerate(zip(features, self.layer_projections)):
            projected = proj(feat)  # [batch, 64]
            fused_features.append(projected)
        
        # 拼接所有层的特征
        concatenated = torch.cat(fused_features, dim=1)  # [batch, 64*4]
        
        return concatenated

    def forward(self, axial_view, coronal_view, sagittal_view):
        """
        axial_view: 轴向视图 [batch, channels, height, width]
        coronal_view: 冠状视图 [batch, channels, height, width]  
        sagittal_view: 矢状视图 [batch, channels, height, width]
        """
        batch_size = axial_view.shape[0]
        
        # 提取各视图的多层特征
        axial_features = self._extract_multi_level_features(axial_view, mode='axial')
        coronal_features = self._extract_multi_level_features(coronal_view, mode='coronal')
        sagittal_features = self._extract_multi_level_features(sagittal_view, mode='sagittal')
        
        # 视图特征融合
        fused_view_features = self.view_fusion(
            axial_features, coronal_features, sagittal_features
        )
        
        # 投影到目标维度
        projected_features = self.feature_proj(fused_view_features)
        
        return projected_features

    # 可选：添加获取各层特征图的方法，用于可视化或进一步处理
    def get_feature_maps(self, x, mode='axial'):
        """获取各层的特征图（不进行池化和投影）"""
        if self.share_weights:
            features = self.backbone(x)
        else:
            if mode == 'axial':
                features = self.axial_backbone(x)
            elif mode == 'coronal':
                features = self.coronal_backbone(x)
            elif mode == 'sagittal':
                features = self.sagittal_backbone(x)
        
        return features  # 返回四层特征图的列表


class ViewFusionModule(nn.Module):
    """三视图特征融合模块"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 注意力权重学习
        self.view_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, batch_first=True
        )
        
        # 门控融合
        self.gate_axial = nn.Linear(feature_dim, feature_dim)
        self.gate_coronal = nn.Linear(feature_dim, feature_dim)
        self.gate_sagittal = nn.Linear(feature_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()
        
        # 可选：添加层归一化
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, axial_feat, coronal_feat, sagittal_feat):
        # 将三个视图特征堆叠 [batch, 3, feature_dim]
        view_features = torch.stack([axial_feat, coronal_feat, sagittal_feat], dim=1)
        
        # 使用注意力机制融合视图特征
        attended_features, _ = self.view_attention(
            view_features, view_features, view_features
        )
        
        # 门控加权融合
        gate_weights = self.sigmoid(
            self.gate_axial(axial_feat) + 
            self.gate_coronal(coronal_feat) + 
            self.gate_sagittal(sagittal_feat)
        )
        
        # 平均池化 + 门控
        mean_features = attended_features.mean(dim=1)  # [batch, feature_dim]
        fused_features = gate_weights * mean_features
        
        # 层归一化
        fused_features = self.layer_norm(fused_features)
        
        return fused_features