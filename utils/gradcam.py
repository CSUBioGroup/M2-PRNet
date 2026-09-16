import torch

# from gradcam_vis import denormalize
class FrozenGradCAM:
    """适用于冻结 backbone，仅基于 feature map 激活生成 CAM"""
    def __init__(self, backbone, target_layer_name):
        self.backbone = backbone
        self.target_layer_name = target_layer_name
        self.activations = None

        # 找到目标层
        self.target_layer = dict(self.backbone.named_modules()).get(target_layer_name, None)
        if self.target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found in backbone")

        # 注册 forward hook
        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        # 直接保存 activations
        self.activations = out.detach()  # [B, C, H, W]
    def __call__(self, x):
        # 前向
        # x = self.denormalize(x)
        _ = self.backbone(x)

        # 计算 pseudo CAM
        act = self.activations  # [B,C,H,W]
        cam = act.mean(dim=1, keepdim=True)  # GAP over channels
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-6
        return cam  # [B,1,H,W]