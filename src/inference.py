from core.gmflow.gmflow import GMFlow
from core.dataset.dataloader import GenerateDataloader, GenerateTestDataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data 
import config
from torchvision.transforms import transforms
from matplotlib.colors import LinearSegmentedColormap
from core.gmflow.geometry import flow_warp
import os
from thop import profile
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time  # 新增计时模块

def DIC_Net_show(output, label, image_h, image_w, vmax_u=None, vmin_u=None, vmax_v=None, vmin_v=None):
    # 初始化 u 和 v
    u = np.zeros((image_h, image_w))
    v = np.zeros((image_h, image_w))
    label_u = np.zeros((image_h, image_w))
    label_v = np.zeros((image_h, image_w))
    
    # 检查 output 格式并赋值
    if isinstance(output, np.ndarray):
        u[:, :] = output[0, 0, :image_h, :image_w]
        v[:, :] = output[0, 1, :image_h, :image_w]
    else:
        raise ValueError("output must be a NumPy array")
    
    if isinstance(label, np.ndarray):
        label_u[:, :] = label[0, 0, :image_h, :image_w]
        label_v[:, :] = label[0, 1, :image_h, :image_w]
    else:
        raise ValueError("label must be a NumPy array")
    
    # 创建绘图区域，2行2列
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    
    # 第一行：u 的 ground truth 和 prediction
    ax1, ax2 = axes[0]
    im1 = ax1.imshow(label_u, vmax=vmax_u, vmin=vmin_u, cmap='jet')
    im2 = ax2.imshow(u, vmax=vmax_u, vmin=vmin_u, cmap='jet')
    
    # 第二行：v 的 ground truth 和 prediction
    ax3, ax4 = axes[1]
    im3 = ax3.imshow(label_v, vmax=vmax_v, vmin=vmin_v, cmap='jet')
    im4 = ax4.imshow(v, vmax=vmax_v, vmin=vmin_v, cmap='jet')
    
    # 设置标题
    ax1.set_title("u ground truth")
    ax2.set_title("u component")
    ax3.set_title("v ground truth")
    ax4.set_title("v component")
    
    # 添加颜色条（每行一个）
    # 第一行颜色条
    divider1 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes("right", size="3%", pad=0.1)
    fig.colorbar(im2, cax=cax1, label="u displacement")
    
    # 第二行颜色条
    divider2 = make_axes_locatable(ax4)
    cax2 = divider2.append_axes("right", size="3%", pad=0.1)
    fig.colorbar(im4, cax=cax2, label="v displacement")
    
    # 显示图像
    plt.show()

def extract_index_from_path(file_path):
    """
    从文件路径中提取序号。
    假设文件名格式为 "dis{序号}.npy" 或 "speckle_{序号}.npy"。
    """
    base_name = os.path.basename(file_path)  # 获取文件名（如 "dis1.npy"）
    file_name, _ = os.path.splitext(base_name)  # 去掉扩展名（如 "dis1"）
    index = ''.join(filter(str.isdigit, file_name))  # 提取数字部分（如 "1"）
    return index


def save_uv1_as_npy(uv1, output_dir, input_file_path):
    """
    将 uv1 保存为 (2, 256, 256) 的 .npy 文件，并以输入文件的序号命名。
    
    参数：
    - uv1: 预测的位移场数据 (NumPy array)，形状为 (1, 2, 256, 256)。
    - output_dir: 输出目录。
    - input_file_path: 输入文件路径（用于提取序号）。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取输入文件的序号
    index = extract_index_from_path(input_file_path)
    
    # 调整 uv1 的形状为 (2, 256, 256)
    uv1_resized = uv1[0]  # 去掉 batch 维度，形状变为 (2, 256, 256)
    
    # 构造输出文件名
    #output_file_name = f"pred_uv_normal_{int(index)}.npy"#####################################################
    #output_file_name = f"pred_uv_ll_{int(index)}.npy"
    output_file_name = f"pred_uv_flowmatching_{int(index)}.npy"

    output_file_path = os.path.join(output_dir, output_file_name)
    
    # 保存为 .npy 文件
    np.save(output_file_path, uv1_resized)
    print(f"Saved prediction to {output_file_path}")


torch.autograd.set_detect_anomaly(True)
opt = config.opt
device = torch.device("cuda")
torch.cuda.empty_cache()
model = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=4,
                   inchannel=3,
                   ).to(device)

model.cuda()

models_path = './models/expert_412.pth'
#models_path = './models/expert_mask.pth'
weights = torch.load(models_path, map_location=device)
model.load_state_dict(weights)
model.eval()

class GMFlowWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, ref_imgs, def_imgs):
        return self.model(
            ref_imgs,
            def_imgs,
            attn_splits_list = [2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
            pred_bidir_flow=False,
            num_reg_refine=6
        )["flow_preds"][-1]

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params / 1e6:.2f} M")
print(f"可训练参数量: {trainable_params / 1e6:.2f} M")

uv_path = "../Data410/uv_256/dis130.npy"
#speckle_path = "../Data410/test/speckle_normal/speckle_130.npy"###################################################################
#speckle_path = "../Data410/test/speckle_ll/speckle_130.npy"
speckle_path = "../Data410/speckle_256/speckle_130.npy"

labels = np.load(uv_path)
labels = torch.unsqueeze(torch.from_numpy(labels), 0).float().to('cuda')
#uv = uv[:, [1, 0], :, :]

speckle = np.load(speckle_path)
speckle = torch.unsqueeze(torch.from_numpy(speckle), 0).float().repeat(1, 3, 1, 1).to('cuda')
#speckle = speckle[:, [1, 0], :, :]

image_h = 256
image_w = 256

ref_imgs = speckle[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
def_imgs = speckle[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)

from thop import profile

wrapped_model = GMFlowWrapper(model).to(device)

wrapped_model.eval()

flops, params = profile(

    wrapped_model,

    inputs=(ref_imgs, def_imgs),

    verbose=False)

print(f"Full MoE FLOPs: {flops / 1e9:.2f} G")
print(f"Full MoE Params: {params / 1e6:.2f} M")

# ========== 模型推理计时核心部分 ==========
# 预热GPU（消除初次cuda上下文延迟，时间更准）
with torch.no_grad():
    _ = model(ref_imgs, def_imgs,
              attn_splits_list=[2],
              corr_radius_list=[-1],
              prop_radius_list=[-1],
              pred_bidir_flow=False,
              num_reg_refine=6)

torch.cuda.synchronize()  # 同步GPU，保证计时准确
start_time = time.time()

with torch.no_grad():  # 推理不用梯度，提速
    pred_flow = model(ref_imgs, def_imgs,
                      attn_splits_list=[2],
                      corr_radius_list=[-1],
                      prop_radius_list=[-1],
                      pred_bidir_flow=False,
                      num_reg_refine=6)

torch.cuda.synchronize()
end_time = time.time()
infer_time_ms = (end_time - start_time) * 1000
print(f"GMFlow 单次模型推理耗时: {infer_time_ms:.2f} ms")
# ==========================================

uv1 = pred_flow['flow_preds'][-1].detach()
# 提取预测的 u 和 v 分量
pred_u = uv1[0, 0, :, :]
pred_v = uv1[0, 1, :, :]
# 获取最大值和最小值
pred_u_max = float(pred_u.max().item())
pred_u_min = float(pred_u.min().item())
pred_v_max = float(pred_v.max().item())
pred_v_min = float(pred_v.min().item())

i = torch.arange(image_h, dtype=torch.float32).to('cuda')  # y坐标
j = torch.arange(image_w, dtype=torch.float32).to('cuda')  # x坐标
ii, jj = torch.meshgrid(i, j, indexing='ij')  

uv1_u = uv1[0, 0, :, :] 
uv1_v = uv1[0, 1, :, :] 

valid_y_uv1 = (ii + uv1_v >= 0) & (ii + uv1_v < image_h)
valid_x_uv1 = (jj + uv1_u >= 0) & (jj + uv1_u < image_w)
mask1 = valid_x_uv1 & valid_y_uv1  # 最终掩码，形状 [image_h, image_w]

mean_flow = torch.mean((uv1*mask1.unsqueeze(0).unsqueeze(0) - labels*mask1.unsqueeze(0).unsqueeze(0)).abs()).item()
print("The value of gmflow uv {}".format(mean_flow))

output_dir = './output'
labels = labels.cpu().numpy()
uv1 = uv1.cpu().numpy()
DIC_Net_show(uv1, labels, image_h, image_w, vmax_u=pred_u_max, vmin_u=pred_u_min, vmax_v=pred_v_max, vmin_v=pred_v_min)
save_uv1_as_npy(uv1, output_dir, uv_path)  # 使用 uv_path 提取序号