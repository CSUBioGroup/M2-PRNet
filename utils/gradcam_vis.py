import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def tensor_to_img(x):
    """把 1×3×H×W tensor 转成 numpy uint8"""
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    x = (x - x.min()) / (x.max() + 1e-6)
    return (x * 255).astype(np.uint8)

def cam_to_heatmap(cam):
    """CAM: 1×1×H×W → heatmap RGB"""
    cam = cam.squeeze().cpu().numpy()
    cam = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.45):
    """RGB 图像融合 CAM 热力图"""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    over = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return over
from matplotlib import cm

# def save_cam_with_colorbar(img_tensor, cam_tensor, save_path,
#                            mean=[0.485, 0.456, 0.406],
#                            std=[0.229, 0.224, 0.225],
#                            gamma=0.5, alpha=0.5):
#     """
#     img_tensor: [3,H,W] 归一化后的输入
#     cam_tensor: [1,H',W'] CAM（0~1）
#     gamma: 对 CAM 做幂次缩放，增强低值对比
#     alpha: CAM 与原图叠加透明度
#     """
#     # -----------------------
#     # 1) 反归一化输入图像
#     # -----------------------
#     mean = torch.tensor(mean).view(3,1,1).to(img_tensor.device)
#     std  = torch.tensor(std).view(3,1,1).to(img_tensor.device)
#     img = img_tensor * std + mean
#     img = img.clamp(0, 1)
#     img = img.permute(1,2,0).cpu().numpy()
#     img_uint8 = (img * 255).astype(np.uint8)

#     # -----------------------
#     # 2) CAM resize + 缩放
#     # -----------------------
#     cam = cam_tensor[0].cpu().numpy()
#     cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
#     cam = cam - cam.min()
#     cam = cam / (cam.max() + 1e-6)
#     cam_scaled = np.power(cam, gamma)   # 非线性缩放

#     # -----------------------
#     # 3) CAM 转为颜色图
#     # -----------------------
#     cmap = cm.get_cmap("turbo")
#     heat_color = (cmap(cam_scaled)[..., :3] * 255).astype(np.uint8)

#     # -----------------------
#     # 4) 叠加到原图
#     # -----------------------
#     overlay = (img_uint8*(1-alpha) + heat_color*alpha).astype(np.uint8)

#     # -----------------------
#     # 5) 保存带颜色条的图像
#     # -----------------------
#     fig, ax = plt.subplots(figsize=(8,8))
#     ax.imshow(overlay)
#     ax.axis('off')

#     # 添加 colorbar
#     norm = plt.Normalize(vmin=0, vmax=1)
#     sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label("CAM intensity", fontsize=12)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
def save_cam_gray_overlay_paper(
    img_tensor,
    cam_tensor,
    save_path,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    cmap_name="turbo",
    alpha=0.8,
    th=0.2,           # CAM 显著阈值
):
    import numpy as np
    import torch
    import cv2
    import matplotlib.cm as cm

    # --------------------------------------------------
    # 1) 反归一化 → RGB
    # --------------------------------------------------
    mean = torch.tensor(mean).view(3,1,1).to(img_tensor.device)
    std  = torch.tensor(std).view(3,1,1).to(img_tensor.device)

    img = (img_tensor * std + mean).clamp(0,1)
    img = img.permute(1,2,0).cpu().numpy()  # [H,W,3]
    H, W = img.shape[:2]

    # --------------------------------------------------
    # 2) 白背景 mask（透明）
    # --------------------------------------------------
    bg_mask = (img > 0.98).all(axis=-1)   # True = 白背景

    # --------------------------------------------------
    # 3) 灰度结构底图
    # --------------------------------------------------
    gray = np.dot(img, [0.299, 0.587, 0.114])
    gray3 = np.stack([gray, gray, gray], axis=-1)

    # --------------------------------------------------
    # 4) CAM resize & normalize
    # --------------------------------------------------
    cam = cam_tensor[0].detach().cpu().numpy()
    cam = cv2.resize(cam, (W, H))

    low, high = np.percentile(cam, [5, 95])
    cam = np.clip(cam, low, high)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    # --------------------------------------------------
    # 5) 显著 CAM mask（软）
    # --------------------------------------------------
    mask = cam.copy()
    mask[mask < th] = 0
    mask = (mask - th) / (1 - th + 1e-6)
    mask = np.clip(mask, 0, 1)

    # --------------------------------------------------
    # 6) 单色系 colormap（论文友好）
    # --------------------------------------------------
    cmap = cm.get_cmap(cmap_name)
    heat = cmap(cam)[:, :, :3]  # RGB

    # --------------------------------------------------
    # 7) 叠加（不发黑、不染背景）
    # --------------------------------------------------
    overlay_rgb = gray3 + heat **1.2 * mask[..., None]
    overlay_rgb = np.clip(overlay_rgb, 0, 1)

    # --------------------------------------------------
    # 8) Alpha：背景透明 + 非显著区域弱化
    # --------------------------------------------------
    
    final_alpha = mask * alpha
    final_alpha[bg_mask] = 0.0

    # --------------------------------------------------
    # 9) RGBA 输出
    # --------------------------------------------------
    out = np.zeros((H, W, 4), dtype=np.float32)
    out[:, :, :3] = overlay_rgb
    out[:, :, 3] = final_alpha

    out_uint8 = (out * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(out_uint8, cv2.COLOR_RGBA2BGRA))
def save_cam_paper_style(img_tensor, cam_tensor, save_path,
                         cmap="YlOrRd",
                         alpha_max=0.75,
                         threshold=0.2,
                         darken_factor=0.5):
    """
    img_tensor: [3, H, W] torch tensor (可以是 GPU)
    cam_tensor: [H, W]    torch tensor (可以是 GPU)
    """

    # -------------------------
    # 0) 转 CPU，再转 numpy，避免报错
    # -------------------------
    img_tensor = img_tensor.detach().cpu().numpy()
    cam_tensor = cam_tensor.detach().cpu().numpy()

    # -------------------------
    # 1) invert normalization
    # -------------------------
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img = img_tensor * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))

    # -------------------------
    # 2) normalize CAM
    # -------------------------
    cam = cam_tensor - cam_tensor.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.astype(np.float32)

    # -------------------------
    # 3) colormap
    # -------------------------
    cmap_fn = cm.get_cmap(cmap)
    heatmap = cmap_fn(cam)[:, :, :3]  # drop alpha
    heatmap = (heatmap * 255).astype(np.uint8)

    # -------------------------
    # 4) background darken
    # -------------------------
    background = (img.astype(np.float32) * darken_factor).astype(np.uint8)

    # -------------------------
    # 5) CAM as alpha
    # -------------------------
    alpha = np.clip((cam - threshold) / (1e-8 + (1 - threshold)), 0, 1)
    alpha = alpha ** 1.5
    alpha = (alpha * alpha_max).astype(np.float32)

    alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)

    # -------------------------
    # 6) final blend
    # -------------------------
    overlay = (heatmap * alpha_3 + background * (1 - alpha_3)).astype(np.uint8)

    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
def save_cam_paper_style_v3(
    img_tensor,
    cam_tensor,
    save_path,
    cmap="YlOrRd",
    focus_gamma=1.8,   # 控制“关注集中度”
):
    import numpy as np
    import torch
    import cv2
    import matplotlib.cm as cm
    from scipy.ndimage import gaussian_filter

    # -----------------------------
    # 1) Image → CPU
    # -----------------------------
    img = img_tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    img = img.permute(1,2,0).numpy()
    img = img.clip(0,1)
    H, W = img.shape[:2]

    # 转灰度（论文风格关键）
    gray = np.dot(img, [0.299, 0.587, 0.114])
    gray = np.stack([gray]*3, axis=-1)

    # -----------------------------
    # 2) CAM → (H,W)
    # -----------------------------
    cam = cam_tensor.detach().cpu()
    while cam.ndim > 2:
        cam = cam.squeeze(0)
    if cam.ndim == 3:
        cam = cam[0]
    cam = cam.numpy()
    cam = cv2.resize(cam, (W, H))

    # robust normalize
    low, high = np.percentile(cam, [10, 95])
    cam = np.clip(cam, low, high)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    # 平滑但不过度
    cam = gaussian_filter(cam, sigma=2)

    # -----------------------------
    # 3) 非线性强调（核心！）
    # -----------------------------
    cam_focus = cam ** focus_gamma   # 拉开差距！

    # -----------------------------
    # 4) colormap
    # -----------------------------
    heat = cm.get_cmap(cmap)(cam_focus)[..., :3]

    # -----------------------------
    # 5) 结构抑制（非重要区域变暗）
    # -----------------------------
    suppress = 0.15 + 0.85 * cam_focus
    suppress_3 = suppress[..., None]

    base = gray * suppress_3

    # -----------------------------
    # 6) CAM 区域直接“贴上去”
    # -----------------------------
    overlay = np.where(
        cam_focus[..., None] > 0.15,
        heat,
        base
    )

    # -----------------------------
    # 7) Alpha：只保留结构区域
    # -----------------------------
    alpha = (gray[...,0] > 0.02).astype(np.float32)

    out = np.zeros((H, W, 4), dtype=np.float32)
    out[..., :3] = overlay
    out[..., 3] = alpha

    out_uint8 = (out * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(out_uint8, cv2.COLOR_RGBA2BGRA))
def save_cam_paper_style_v2(
    img_tensor,
    cam_tensor,
    save_path,
    cmap="YlOrRd",
    alpha=0.55,       # CAM 覆盖强度
    smooth=True,      # CAM 平滑
):
    import numpy as np
    import torch
    import cv2
    import matplotlib.cm as cm
    from scipy.ndimage import gaussian_filter

    # -----------------------------------
    # 1) 修正 img_tensor → CPU + squeeze
    # -----------------------------------
    img = img_tensor.detach().cpu()
    if img.ndim == 4: img = img[0]
    img = img.permute(1,2,0).numpy()   # (H,W,3)
    img = img.clip(0,1)

    H, W = img.shape[:2]

    # -----------------------------------
    # 2) 强制把 CAM → (H,W)
    # -----------------------------------
    cam = cam_tensor.detach().cpu()

    while cam.ndim > 2:
        cam = cam.squeeze(0)

    if cam.ndim == 3:   # (C,H,W) 取通道 0
        cam = cam[0]

    cam = cam.numpy()

    # resize 保证与图一致
    cam = cv2.resize(cam, (W, H))

    # -----------------------------------
    # 3) 正规化 + 平滑
    # -----------------------------------
    # 去掉极值
    low, high = np.percentile(cam, [5, 95])
    cam = np.clip(cam, low, high)

    # 归一化到 0-1
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    # 高斯平滑，使边界更自然（论文风格）
    if smooth:
        cam = gaussian_filter(cam, sigma=4)

    # -----------------------------------
    # 4) YlOrRd colormap
    # -----------------------------------
    cmap_func = cm.get_cmap(cmap)
    heat = cmap_func(cam)[..., :3]   # 只取 RGB

    # -----------------------------------
    # 5) Alpha mask（透明度随 cam 强弱变化）
    # -----------------------------------
    alpha_map = cam * alpha
    alpha_3 = np.repeat(alpha_map[..., None], 3, axis=2)  # (H,W,3)

    # -----------------------------------
    # 6) 原图正片叠底（保留结构线条）
    # -----------------------------------
    # 增加对比度，类似 PyMOL cartoon 质感
    img_enhance = np.power(img, 0.85)

    overlay = heat * alpha_3 + img_enhance * (1 - alpha_3)

    # -----------------------------------
    # 7) 背景透明 (cam < eps)
    # -----------------------------------
    eps = 0.03
    alpha_final = (cam > eps).astype(np.float32)

    # -----------------------------------
    # 8) 写入 RGBA PNG
    # -----------------------------------
    out = np.zeros((H, W, 4), dtype=np.float32)
    out[..., :3] = overlay
    out[..., 3] = alpha_final

    out_uint8 = (out * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(out_uint8, cv2.COLOR_RGBA2BGRA))
# def save_cam_gray_overlay_paper(
#     img_tensor,
#     cam_tensor,
#     save_path,
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225],
#     cmap_name="viridis",   # IJCAI 非常友好
#     alpha=0.6,
#     gamma=0.8,             # < 1 让中低响应可见
# ):
#     """
#     IJCAI-style CAM visualization:
#     - cool monochrome base
#     - perceptually uniform colormap
#     - restrained contrast
#     - clean background
#     """

#     import numpy as np
#     import torch
#     import cv2
#     import matplotlib.cm as cm

#     # --------------------------------------------------
#     # 1. Denormalize → RGB
#     # --------------------------------------------------
#     mean = torch.tensor(mean, device=img_tensor.device).view(3,1,1)
#     std  = torch.tensor(std,  device=img_tensor.device).view(3,1,1)

#     img = (img_tensor * std + mean).clamp(0, 1)
#     img = img.permute(1, 2, 0).detach().cpu().numpy()
#     H, W = img.shape[:2]

#     # --------------------------------------------------
#     # 2. White background → transparent
#     # --------------------------------------------------
#     bg_mask = (img > 0.97).all(axis=-1)

#     # --------------------------------------------------
#     # 3. Monochrome base (cool gray-blue)
#     # --------------------------------------------------
#     gray = np.dot(img, [0.299, 0.587, 0.114])
#     base_color = np.array([0.78, 0.82, 0.87])   # IJCAI-safe
#     base = gray[..., None] * base_color

#     # --------------------------------------------------
#     # 4. CAM resize & robust normalize
#     # --------------------------------------------------
#     cam = cam_tensor[0].detach().cpu().numpy()
#     cam = cv2.resize(cam, (W, H))

#     low, high = np.percentile(cam, [2, 98])
#     cam = np.clip(cam, low, high)
#     cam = (cam - low) / (high - low + 1e-6)

#     cam_vis = cam ** gamma

#     # --------------------------------------------------
#     # 5. Color mapping (no saturation)
#     # --------------------------------------------------
#     cmap = cm.get_cmap(cmap_name)
#     heat = cmap(cam_vis)[..., :3]

#     # mild desaturation (very important for IJCAI)
#     heat = 0.85 * heat + 0.15 * base

#     # --------------------------------------------------
#     # 6. Alpha mapping (linear & restrained)
#     # --------------------------------------------------
#     alpha_map = alpha * cam_vis

#     # --------------------------------------------------
#     # 7. Blend
#     # --------------------------------------------------
#     out_rgb = (
#         base * (1 - alpha_map[..., None]) +
#         heat * alpha_map[..., None]
#     )

#     # --------------------------------------------------
#     # 8. Final alpha channel
#     # --------------------------------------------------
#     final_alpha = (~bg_mask).astype(np.float32)

#     # --------------------------------------------------
#     # 9. RGBA output
#     # --------------------------------------------------
#     out = np.zeros((H, W, 4), dtype=np.float32)
#     out[..., :3] = out_rgb
#     out[..., 3]  = final_alpha

#     out = (out * 255).astype(np.uint8)
#     cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))
from matplotlib.colors import LinearSegmentedColormap

def make_hue_spectrum_cmap():
    """在HSL色彩空间中，色调值从约240度（蓝色）渐变到0度（红色）"""
    from matplotlib.colors import LinearSegmentedColormap
    cdict = {
        'red':   [(0.0, 0.2, 0.2),   # 蓝色端，红色分量很少
                  (0.5, 0.0, 0.0),   # 中间（青色/品红过渡区），红色分量可设0
                  (1.0, 0.86, 0.86)],# 红色端，红色分量最高
        'green': [(0.0, 0.4, 0.4),   # 蓝色端，绿色分量中等
                  (0.5, 0.0, 0.0),   # 中间，绿色分量可设0
                  (1.0, 0.28, 0.28)],# 红色端，绿色分量较低
        'blue':  [(0.0, 0.72, 0.72), # 蓝色端，蓝色分量最高
                  (0.5, 1.0, 1.0),   # 中间，蓝色分量可最高（会呈现品红/紫色）
                  (1.0, 0.21, 0.21)] # 红色端，蓝色分量很低
    }
    # 注：上方0.5位置的参数可以调整，以控制中间过渡色是紫色、品红还是白色。
    # 如果想避免中间出现强烈的紫色/品红，可以将0.5点的RGB设为(0.95, 0.95, 0.95)接近白色。
    return LinearSegmentedColormap('hue_spectrum', cdict)
def make_ijcai_warm_cmap():
    colors = [
        # (0.00, (0.96, 0.94, 0.90)),  # almost beige
        # (0.25, (0.99, 0.88, 0.65)),  # light sand
        (0.00, (253/255.0,229/255.0,201/255.0)),  # soft orange
        (0.50, (0.86, 0.28, 0.21)),  # muted red
        (1.00, (0.55, 0.05, 0.07)),  # deep highlight
    ]
    return LinearSegmentedColormap.from_list("ijcai_warm", colors)
def make_cool_to_warm_cmap():
    """从蓝（冷）过渡到橙/红（热）"""
    colors = [
        # (0.0, (0.22, 0.49, 0.72)),   # 起点：沉稳的蓝色
        (0.0, (0.65, 0.81, 0.89)),   # 过渡：浅蓝色
        (0.33, (0.90, 0.90, 0.90)),   # 中间：中性白/浅灰，用于平滑衔接
        (0.67, (0.98, 0.74, 0.50)),   # 过渡：浅橙色
        (1.0, (0.86, 0.28, 0.21)),   # 终点：您的主题红色
    ]
    return LinearSegmentedColormap.from_list("cool_to_warm", colors)
def save_cam_gray_overlay_paper(
    img_tensor,
    cam_tensor,
    save_path,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    cmap_name="viridis",
    alpha=0.6,
    gamma=0.8,
):
    """
    IJCAI-style CAM (warm beige base)
    """

    import numpy as np
    import torch
    import cv2
    import matplotlib.cm as cm

    # 1. Denormalize
    mean = torch.tensor(mean, device=img_tensor.device).view(3,1,1)
    std  = torch.tensor(std,  device=img_tensor.device).view(3,1,1)

    img = (img_tensor * std + mean).clamp(0, 1)
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    H, W = img.shape[:2]

    # 2. Background mask
    bg_mask = (img > 0.97).all(axis=-1)

    # 3. Warm beige base
    gray = np.dot(img, [0.299, 0.587, 0.114])
    base_color = np.array([0.90, 0.88, 0.82])   # ★ 核心修改
    base = gray[..., None] * base_color

    # 4. CAM normalize
    cam = cam_tensor[0].detach().cpu().numpy()
    cam = cv2.resize(cam, (W, H))
    # Normalize (tailored for your distribution)
    low, high = np.percentile(cam, [5, 95])
    cam = np.clip(cam, low, high)
    cam = (cam - low) / (high - low + 1e-6)

    # Contrast shaping
    cam_vis = cam ** 1.4

    
    # low, high = np.percentile(cam, [2, 98])
    # cam = np.clip(cam, low, high)
    # cam = (cam - low) / (high - low + 1e-6)
    # cam_vis = cam ** gamma
    # print(
    # f"cam stats: min={cam.min():.3f}, "
    # f"p10={np.percentile(cam,10):.3f}, "
    # f"p50={np.percentile(cam,50):.3f}, "
    # f"p90={np.percentile(cam,90):.3f}, "
    # f"max={cam.max():.3f}"
    # )
    # 5. Colormap
    # cmap = cm.get_cmap(cmap_name)
    cmap = make_ijcai_warm_cmap()
    heat = cmap(cam_vis)[..., :3]

    # mild desaturation (very IJCAI)
    heat = 0.8 * heat + 0.2 * base

    # 6. Alpha
    # alpha_map = alpha * cam_vis
    # Alpha shaping
    alpha_map = 0.55 * np.sqrt(cam_vis)

    # 7. Blend
    out_rgb = (
        base * (1 - alpha_map[..., None]) +
        heat * alpha_map[..., None]
    )
    out_rgb = heat
    # 8. Alpha channel
    final_alpha = (~bg_mask).astype(np.float32)

    # 9. Save RGBA
    out = np.zeros((H, W, 4), dtype=np.float32)
    out[..., :3] = out_rgb
    out[..., 3]  = final_alpha

    out = (out * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))
def save_cam_gray_overlay(
    img_tensor,
    cam_tensor,
    save_path,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    colormap="coolwarm",
    alpha=0.6,
):
    import numpy as np
    import torch
    import cv2
    import matplotlib.cm as cm

    # ------------------------------------------------------
    # 1) 反归一化 → RGB 图
    # ------------------------------------------------------
    mean = torch.tensor(mean).reshape(3,1,1).to(img_tensor.device)
    std = torch.tensor(std).reshape(3,1,1).to(img_tensor.device)
    img = (img_tensor * std + mean).clamp(0,1)
    img = img.permute(1,2,0).cpu().numpy()  # [H,W,3]
    H, W = img.shape[:2]

    # ------------------------------------------------------
    # 2) 白色背景 mask（自动检测）
    # ------------------------------------------------------
    bg_mask = (img > 0.98).all(axis=-1)    # True=背景（白）

    # ------------------------------------------------------
    # 3) 灰度底图
    # ------------------------------------------------------
    gray = np.dot(img, [0.299, 0.587, 0.114])
    gray3 = np.stack([gray, gray, gray], axis=-1)

    # ------------------------------------------------------
    # 4) CAM → 拉伸 & 归一化
    # ------------------------------------------------------
    cam = cam_tensor[0].cpu().numpy()
    cam = cv2.resize(cam, (W, H))

    low, high = np.percentile(cam, [5, 95])
    cam = np.clip(cam, low, high)
    cam_trans_mask = cam > 0.99
    
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    # cam +=0.5
    # cam = np.clip(cam, 0, high)
    # CAM 高的区域（0.9以上）强制透明
    # cam_trans_mask = cam > 0.9

    # ------------------------------------------------------
    # 5) 颜色映射
    # ------------------------------------------------------
    cmap = cm.get_cmap(colormap)
    heat = cmap(cam)[:, :, :3]  # RGB only

    # ------------------------------------------------------
    # 6) α 映射（CAM 小更显眼）
    # ------------------------------------------------------
    alpha_map = alpha * (cam)  # CAM 大透明；小不透明

    # ------------------------------------------------------
    # 7) 叠加：灰度 + 热力图
    # ------------------------------------------------------
    out_rgb = heat * alpha_map[..., None] + gray3 * (1 - alpha_map[..., None])
    # out_rgb = heat + gray3
    # ------------------------------------------------------
    # 8) 最终透明度：背景透明 & CAM 高透明
    # ------------------------------------------------------
    final_alpha = (~bg_mask & ~cam_trans_mask).astype(np.float32)

    # ------------------------------------------------------
    # 9) 合成 RGBA
    # ------------------------------------------------------
    out = np.zeros((H, W, 4), dtype=np.float32)
    out[:, :, :3] = out_rgb
    out[:, :, 3] = final_alpha

    # ------------------------------------------------------
    # 10) 保存 PNG (RGBA)
    # ------------------------------------------------------
    out_uint8 = (out * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(out_uint8, cv2.COLOR_RGBA2BGRA))
def save_cam_gray_overlay_clean(
    img_tensor,
    cam_tensor,
    save_path,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    color=(0.8, 0.2, 0.2),  # 单色：暗红（论文最常用）
    alpha=0.7,
):
    import numpy as np
    import torch
    import cv2

    # ------------------------------------------------------
    # 1) 反归一化 → RGB
    # ------------------------------------------------------
    mean = torch.tensor(mean).reshape(3,1,1).to(img_tensor.device)
    std = torch.tensor(std).reshape(3,1,1).to(img_tensor.device)
    img = (img_tensor * std + mean).clamp(0,1)
    img = img.permute(1,2,0).cpu().numpy()
    H, W = img.shape[:2]

    # ------------------------------------------------------
    # 2) 白背景 mask
    # ------------------------------------------------------
    bg_mask = (img > 0.98).all(axis=-1)

    # ------------------------------------------------------
    # 3) 灰度底图（略微压暗）
    # ------------------------------------------------------
    gray = np.dot(img, [0.299, 0.587, 0.114])
    gray = gray * 0.85
    gray3 = np.stack([gray, gray, gray], axis=-1)

    # ------------------------------------------------------
    # 4) CAM → 归一化
    # ------------------------------------------------------
    cam = cam_tensor.detach().cpu()
    if cam.ndim == 3:
        cam = cam[0]
    cam = cam.numpy()
    cam = cv2.resize(cam, (W, H))

    low, high = np.percentile(cam, [1, 99])
    cam = np.clip(cam, low, high)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    # ------------------------------------------------------
    # 5) 单色 CAM（强度控制）
    # ------------------------------------------------------
    cam_strength = (cam ** 1.1) * alpha   # 拉开对比
    cam_color = np.zeros_like(gray3)
    cam_color[..., 0] = color[0]
    cam_color[..., 1] = color[1]
    cam_color[..., 2] = color[2]

    # ------------------------------------------------------
    # 6) 融合（关键）
    # ------------------------------------------------------
    out_rgb = gray3 * (1 - cam_strength[..., None]) \
              + cam_color * cam_strength[..., None]

    # ------------------------------------------------------
    # 7) 透明度
    # ------------------------------------------------------
    final_alpha = (~bg_mask).astype(np.float32)

    # ------------------------------------------------------
    # 8) RGBA 输出
    # ------------------------------------------------------
    out = np.zeros((H, W, 4), dtype=np.float32)
    out[..., :3] = out_rgb
    out[..., 3] = final_alpha

    out_uint8 = (out * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(out_uint8, cv2.COLOR_RGBA2BGRA))
def save_cam_better(
    img_tensor,
    cam_tensor,
    save_path,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    colormap="coolwarm",
    alpha=0.8,
):
    """
    增强版 CAM 可视化：
        - 原图中接近 (0,0,0) 的区域自动恢复为透明
        - CAM 越大越透明
        - 透明区域不叠加热力图
        - 输出 RGBA PNG
    """
    import numpy as np
    import torch
    import cv2
    import matplotlib.cm as cm

    # -----------------------
    # 1) 反归一化图像
    # -----------------------
    mean = torch.tensor(mean).reshape(3,1,1).to(img_tensor.device)
    std = torch.tensor(std).reshape(3,1,1).to(img_tensor.device)
    img = (img_tensor * std + mean).clamp(0,1)
    img = img.permute(1,2,0).cpu().numpy()   # [H,W,3]
    H, W = img.shape[:2]

    # -----------------------
    # 2) 恢复透明区域（黑背景）
    # -----------------------
    # 阈值可根据你的数据调整
    th = 10/255.0
    transparent_mask = (img.mean(axis=-1) < th).astype(np.float32)  # 1=原黑背景 → 应透明

    orig_alpha = 1 - transparent_mask  # 透明区域=0, 结构区域=1

    # 把透明区域 RGB 全部置为 0，避免叠加热力图污染
    img = img * orig_alpha[..., None]

    # -----------------------
    # 3) CAM 处理
    # -----------------------
    cam = cam_tensor[0].cpu().numpy()
    cam = cv2.resize(cam, (W, H))

    low, high = np.percentile(cam, [5, 40])
    cam = np.clip(cam, low, high)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    # -----------------------
    # 4) colormap → RGBA
    # -----------------------
    cmap = cm.get_cmap(colormap)
    heat = cmap(cam)  # [H, W, 4]

    # CAM 越大 heat 越透明
    cam_alpha = alpha * (1 - cam)

    # 应用 alpha
    heat[:, :, 3] = cam_alpha

    # -----------------------
    # 5) 叠加逻辑（最关键）
    # -----------------------
    out = np.zeros((H, W, 4), dtype=np.float32)

    # 结构区域：正常叠加 heatmap
    out[:, :, :3] = heat[:, :, :3] * cam_alpha[..., None] + img * (1 - cam_alpha[..., None])

    # alpha：原图透明区域保持透明
    out[:, :, 3] = orig_alpha

    # -----------------------
    # 6) 保存 RGBA PNG
    # -----------------------
    out_uint8 = (out * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(out_uint8, cv2.COLOR_RGBA2BGRA))
# def save_cam_better(
#     img_tensor,
#     cam_tensor,
#     save_path,
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225],
#     colormap="coolwarm",     # jet/coolwarm/turbo
#     alpha=0.5,         # 叠加透明度
# ):
#     """
#     更漂亮、更有区分度的 CAM 可视化函数。
#     """

#     # -----------------------
#     # 1) 反归一化图像
#     # -----------------------
#     mean = torch.tensor(mean).reshape(3,1,1).to(img_tensor.device)
#     std = torch.tensor(std).reshape(3,1,1).to(img_tensor.device)
#     img = (img_tensor * std + mean).clamp(0,1)
#     img = img.permute(1,2,0).cpu().numpy()
#     img_uint8 = (img*255).astype(np.uint8)
#     gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)  # [H,W]
#     gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)        # [H,W,3]
#     # -----------------------
#     # 2) 处理 CAM
#     # -----------------------
#     cam = cam_tensor[0].cpu().numpy()
#     cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

#     low, high = np.percentile(cam, [5, 40])
#     cam = np.clip(cam, low, high)
#     cam = cam - cam.min()
#     cam = cam / (cam.max() + 1e-6)
#     # cam = 1 - cam 
#     # -----------------------
#     # 3) Colormap
#     # -----------------------
#     cmap = cm.get_cmap(colormap)
    
#     heatmap_rgba = cmap(cam)   # 关
#     transparency_mask = cam >= (1.0 - 1e-6)
#     heatmap_rgba[:, :, 3] = np.where(transparency_mask, 0.0, alpha)
#     # gray[:, :, 3] = np.where(transparency_mask, 0.0, alpha)
#     # -----------------------
#     # 5) 保存或叠加
#     # -----------------------
#     # 方案1：直接保存RGBA图像（推荐，最简单）
#     plt.imsave(save_path.replace('.png', '_rgba.png'), heatmap_rgba)

#     # heatmap = cmap(cam)[:, :, :3]   # 去掉 alpha 通道
#     heatmap = cmap(cam)
#     heatmap[:, :, 3] = np.where(cam >= 1.0 - 1e-6, 0.0, alpha)
#     heatmap[:,:,3] = np.where(transparency_mask, 0.0, alpha)
#     heatmap = (heatmap * 255).astype(np.uint8)
    
#     # -----------------------
#     # 4) 更自然的叠加方式（透明度）
#     # -----------------------
#     # heatmap_rgb = heatmap[:, :, :3]  # 取RGB部分，[H,W,3]
#     # 将单通道灰度图扩展为三通道，以便与heatmap_rgb叠加
    
#     overlay = cv2.addWeighted(gray, 1 - alpha, heatmap_rgba, alpha, 0)  # 正确叠加
#     # overlay = cv2.addWeighted(gray, 1 - alpha, heatmap, alpha, 0)

#     # -----------------------
#     # 5) 保存图像 + 高级色条
#     # -----------------------
#     H, W, _ = overlay.shape
#     fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
#     ax.imshow(overlay)
#     ax.axis("off")

#     # 右侧色条
#     cax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
#     norm = plt.Normalize(vmin=cam.min(), vmax=cam.max())
#     cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax)
#     cb.ax.tick_params(labelsize=10)

#     plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
#     plt.close()
def save_cam(img_tensor, cam_tensor, save_path,
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]):
    """
    img_tensor: [3,H,W] 归一化后的输入
    cam_tensor: [1,H',W'] CAM（0~1）
    该函数将生成的色带改为「叠加后颜色」色条（即 colormap 与灰度底图叠加后的颜色），
    便于直观表示叠加区的实际颜色。
    """
    colorbar_width=0.05
    # -----------------------
    # 1) 反归一化输入图像
    # -----------------------
    mean = torch.tensor(mean).view(3,1,1).to(img_tensor.device)
    std  = torch.tensor(std).view(3,1,1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = img.clamp(0, 1)

    # [3,H,W] → [H,W,3]
    img = img.permute(1,2,0).cpu().numpy()
    img_uint8 = (img * 255).astype(np.uint8)

    # -----------------------
    # 2) 转为高亮灰度图
    # -----------------------
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)  # [H,W]
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)        # [H,W,3]
    
    # -----------------------
    # 3) CAM resize
    # -----------------------
    cam = cam_tensor[0].cpu().numpy()
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    # -----------------------
    # 4) 强激活区域 mask
    # -----------------------
    th = 0.2    # 可调
    mask = cam.copy()
    mask[mask < th] = 0
    mask = (mask - th) / (1 - th + 1e-6)
    mask = np.clip(mask, 0, 1)
    mask = mask[..., None]   # [H,W,1]

    # -----------------------
    # 5) TURBO colormap（亮、无蓝色）
    # -----------------------
    heat = (cam * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    # -----------------------
    # 6) 在灰度图上叠加显著区域（对比极强）
    # -----------------------
    overlay = gray.astype(float)
    overlay += heat_color * mask
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # -----------------------
    # 色带设置 & 保存（显示「叠加后颜色」）
    # -----------------------
    H, W, _ = overlay.shape
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.axis('off')
    ax.imshow(overlay)

    # 计算有效区间（基于非零像素或 2/99 百分位截断）
    flat = cam.ravel()
    nonzero = flat[flat > 0]
    if nonzero.size > 0:
        vmin, vmax = np.percentile(nonzero, [2, 99])
    else:
        vmin, vmax = np.percentile(flat, [2, 99])
    if vmax - vmin < 1e-6:
        vmin = max(0.0, vmin - 1e-3)
        vmax = min(1.0, vmax + 1e-3)

    # 准备用于色条的渐变值
    n_steps = 256
    vals = np.linspace(vmin, vmax, n_steps)

    # 背景用灰度图的全局平均值（近似叠加时的底色）
    bg_mean = gray.reshape(-1, 3).mean(axis=0)  # [3], 0..255

    # 生成每个 vals 对应的叠加后颜色（按 overlay 的叠加规则 overlay = gray + heat_color*mask）
    cmap = cm.get_cmap('turbo')
    colorbar_colors = np.zeros((n_steps, 1, 3), dtype=np.uint8)
    for i, v in enumerate(vals):
        # 对应 heat_color
        heat_rgb = (np.array(cmap(v))[:3] * 255.0).astype(np.float32)  # [3]
        # mask 对应值（与主图保持一致的阈值逻辑）
        m = 0.0 if v < th else (v - th) / (1 - th + 1e-6)
        # overlay_color = bg_mean + heat_rgb * m
        overlay_rgb = bg_mean.astype(np.float32) + heat_rgb * m
        overlay_rgb = np.clip(overlay_rgb, 0, 255).astype(np.uint8)
        colorbar_colors[i, 0, :] = overlay_rgb

    # 在右侧新建轴显示 colorbar_colors（垂直）
    cax = fig.add_axes([1-colorbar_width, 0, colorbar_width, 1])
    cax.imshow(colorbar_colors, aspect='auto')
    cax.yaxis.set_ticks([0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1])
    # 将 ticks 转为对应的 cam 值并格式化
    tick_vals = np.linspace(vmin, vmax, 5)
    cax.set_yticklabels([f"{t:.2f}" for t in tick_vals])
    cax.set_xticks([])
    cax.invert_yaxis()  # 使小值在上（可根据习惯调整）

    plt.subplots_adjust(left=0, right=1-colorbar_width, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
# def save_cam(img_tensor, cam_tensor, save_path,
#              mean=[0.485, 0.456, 0.406],
#              std=[0.229, 0.224, 0.225]):
#     """
#     img_tensor: [3,H,W] 归一化后的输入
#     cam_tensor: [1,H',W'] CAM（0~1）
#     """
#     colorbar_width=0.05
#     # -----------------------
#     # 1) 反归一化输入图像
#     # -----------------------
#     mean = torch.tensor(mean).view(3,1,1).to(img_tensor.device)
#     std  = torch.tensor(std).view(3,1,1).to(img_tensor.device)
#     img = img_tensor * std + mean
#     img = img.clamp(0, 1)

#     # [3,H,W] → [H,W,3]
#     img = img.permute(1,2,0).cpu().numpy()
#     img_uint8 = (img * 255).astype(np.uint8)

#     # -----------------------
#     # ⭐ 2) 转为高亮灰度图
#     # -----------------------
#     gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)  # [H,W]
#     gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)        # [H,W,3]
    
#     # -----------------------
#     # 3) CAM resize
#     # -----------------------
#     cam = cam_tensor[0].cpu().numpy()
#     cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

#     cam = cam - cam.min()
#     cam = cam / (cam.max() + 1e-6)

#     # -----------------------
#     # 4) 强激活区域 mask
#     # -----------------------
#     th = 0.2    # 可调
#     mask = cam.copy()
#     mask[mask < th] = 0
#     mask = (mask - th) / (1 - th + 1e-6)
#     mask = np.clip(mask, 0, 1)
#     mask = mask[..., None]   # [H,W,1]

#     # -----------------------
#     # 5) TURBO colormap（亮、无蓝色）
#     # -----------------------
#     heat = (cam * 255).astype(np.uint8)
#     heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
#     heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

#     # -----------------------
#     # ⭐ 6) 在灰度图上叠加显著区域（对比极强）
#     # -----------------------
#     overlay = gray.astype(float)
#     overlay += heat_color * mask
#     overlay = np.clip(overlay, 0, 255).astype(np.uint8)
#     H, W, _ = overlay.shape
#     fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
#     ax.axis('off')
#     ax.imshow(overlay)

#     # colorbar：仅展示有效区间（基于非零像素或 2/99 百分位截断）
#     cax = fig.add_axes([1-colorbar_width, 0, colorbar_width, 1])
#     flat = cam.ravel()
#     nonzero = flat[flat > 0]
#     if nonzero.size > 0:
#         vmin, vmax = np.percentile(nonzero, [2, 99])
#     else:
#         # 若没有非零，使用全域小范围
#         vmin, vmax = np.percentile(flat, [2, 99])
#     if vmax - vmin < 1e-6:
#         # 扩展微小区间，避免除零
#         vmin = max(0.0, vmin - 1e-3)
#         vmax = min(1.0, vmax + 1e-3)
#     cmap = cm.get_cmap('turbo')
#     norm = plt.Normalize(vmin=vmin, vmax=vmax)
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cb = plt.colorbar(sm, cax=cax)
#     # 可选：只显示几档刻度
#     cb_ticks = np.linspace(vmin, vmax, num=5)
#     cb.set_ticks(cb_ticks)
#     cb.set_ticklabels([f"{t:.2f}" for t in cb_ticks])
#     plt.subplots_adjust(left=0, right=1-colorbar_width, top=1, bottom=0)
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

    # colorbar
    # cax = fig.add_axes([1-colorbar_width, 0, colorbar_width, 1])
    # norm = plt.Normalize(vmin=cam.min(), vmax=cam.max())
    # sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, cax=cax)
    # plt.subplots_adjust(left=0, right=1-colorbar_width, top=1, bottom=0)
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.close()
    # # 色带设置
    # cax = fig.add_axes([1-colorbar_width, 0, colorbar_width, 1])
    # norm = plt.Normalize(vmin=cam.min(), vmax=cam.max())
    # sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, cax=cax)
    # # -----------------------
    # # 7) 保存
    # # -----------------------
    # cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
def save_cam_with_colorbar(img_tensor, cam_tensor, save_path,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225],
                           gamma=0.5, alpha=0.7, th=0.2,
                           colorbar_width=0.05):
    """
    img_tensor: [3,H,W] 归一化后的输入
    cam_tensor: [1,H',W'] CAM（0~1）
    gamma: CAM 对数/幂次缩放，增强低值差异
    alpha: CAM 叠加权重
    th: CAM 阈值，小于该值部分淡化
    colorbar_width: 色带占图片宽度比例
    """
    # -----------------------
    # 1) 反归一化输入
    # -----------------------
    mean = torch.tensor(mean).view(3,1,1).to(img_tensor.device)
    std  = torch.tensor(std).view(3,1,1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1,2,0).cpu().numpy()
    img_uint8 = (img * 255).astype(np.uint8)

    # -----------------------
    # 2) 灰度底图
    # -----------------------
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # -----------------------
    # 3) CAM resize + 归一化 + gamma 缩放
    # -----------------------
    cam = cam_tensor[0].cpu().numpy()
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    cam_scaled = np.power(cam, gamma)

    # -----------------------
    # 4) 生成 mask（空白区域淡化）
    # -----------------------
    mask = np.clip((cam_scaled - th) / (1 - th + 1e-6), 0, 1)[..., None]

    # -----------------------
    # 5) CAM 转颜色图
    # -----------------------
    heat_color = (cm.get_cmap("turbo")(cam_scaled)[..., :3] * 255).astype(np.uint8)

    # -----------------------
    # 6) 叠加 CAM（空白部分淡化）
    # -----------------------
    overlay = gray.astype(float) * (1 - mask) + heat_color.astype(float) * mask * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # -----------------------
    # 7) 绘制色带
    # -----------------------
    H, W, _ = overlay.shape
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.axis('off')
    ax.imshow(overlay)

    # 色带设置
    cax = fig.add_axes([1-colorbar_width, 0, colorbar_width, 1])
    norm = plt.Normalize(vmin=cam.min(), vmax=cam.max())
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax)

    # 保存
    plt.subplots_adjust(left=0, right=1-colorbar_width, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
def denormalize(tensor, mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]):
    """
    将 Normalize 过的图像恢复到原始像素空间（0~1）
    tensor: [3,H,W]
    """
    mean = torch.tensor(mean).view(3,1,1).to(tensor.device)
    std = torch.tensor(std).view(3,1,1).to(tensor.device)
    return tensor * std + mean