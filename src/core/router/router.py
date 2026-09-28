import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from core.gmflow.backbone import CNNEncoderRouter
except ImportError:
    from flowformer_1209.gmflow.backbone import CNNEncoderRouter

try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError:
    spatial_correlation_sample = None
from core.gmflow.geometry import coords_grid, generate_window_grid, normalize_coords

def coords_grid(b, h, w, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # [H, W]
    stacks = [x, y]
    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros'):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


    return img

def global_correlation_softmax(feature0, feature1,
                               pred_bidir_flow=False,
                              
                               ):

    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  # [B, H, W, H, W]
    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]
    prob = F.softmax(correlation, dim=-1).view(b, h*w, h, w)# [B, H*W, H*W]

    grid = coords_grid(b, 16, 16, device=feature0.device)
    prob = bilinear_sample(prob, grid)
    
    return prob.squeeze(1).view(b, h*w, 16*16).permute(0, 2, 1).view(b, 16*16, h, w)

def local_correlation_softmax(feature0, feature1, local_radius=4,padding_mode='zeros'):

        b, c, h, w = feature0.size()
        coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
        coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1
        window_grid = generate_window_grid(-local_radius, local_radius,
                                        -local_radius, local_radius,
                                        local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
        sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]
        sample_coords_softmax = sample_coords
        # exclude coords that are out of image space
        valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
        valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]
        valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax
        # normalize coordinates to [-1, 1]
        sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
        window_feature = F.grid_sample(feature1.contiguous(), sample_coords_norm.contiguous(),
                                    padding_mode=padding_mode, align_corners=False
                                    ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
        feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

        corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

        corr[~valid] = -1e9
        corr = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]
        

        correspondence = torch.matmul(corr.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
                b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
        corr = corr.permute(0, 2, 1).view(b,(2*local_radius+1)*(2*local_radius+1), h, w )
        flow = correspondence - coords_init

        return flow, corr


class Router(nn.Module):
    def __init__(self, opt):
        super(Router, self).__init__()

        self.backbone = CNNEncoderRouter(output_dim=128, norm_layer=nn.InstanceNorm2d, num_output_scales=1, inchannel=3)
        self.conv1 = nn.Sequential(nn.Conv2d(15*15, 128, kernel_size=3, stride=1, padding=1),
                                   torch.nn.LeakyReLU(negative_slope=0.1, inplace=False),
                                   nn.Conv2d(128, 128, 3, 1, 1),
                                   torch.nn.LeakyReLU(negative_slope=0.1, inplace=False),
                                   #nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.BatchNorm2d(256),
                                   torch.nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   torch.nn.LeakyReLU(negative_slope=0.1, inplace=False),
                                   nn.Conv2d(256, 256, 3, 1, 1),
                                   torch.nn.LeakyReLU(negative_slope=0.1, inplace=False),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.BatchNorm2d(512),
                                   torch.nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                   torch.nn.LeakyReLU(negative_slope=0.1, inplace=False),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(512, 512, 3, 1, 1),
                                   torch.nn.LeakyReLU(negative_slope=0.1, inplace=False),
                                   nn.Conv2d(512, 1024, 3, 1, 1))
        self.linear = nn.Sequential(torch.nn.Linear(1024, 1024),
                                    torch.nn.LeakyReLU(negative_slope=0.1, inplace=False),
                                    torch.nn.Linear(1024,512),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(512,1),
                                    torch.nn.Sigmoid())

    def forward(self, img0, img1):
        
        b,c,h,w = img0.shape
        if img0.shape[1] == 1:
            img0 = img0.repeat(1, 3, 1, 1)
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)

        concat = torch.cat((img0, img1), dim=0)
        features = self.backbone(concat)
        chunks = torch.chunk(features, 2, 0)  # tuple
        feature0 = chunks[0]
        feature1 = chunks[1]            

        #grid_img = coords_grid(b, h, w, device=img0.device)#[B,2,H,W]
        #standard_grid = coords_grid(b, 32, 32, device=img0.device)
        #grid = bilinear_sample(grid_img, standard_grid, mode='bilinear', padding_mode='zeros').repeat(1,16,1,1)#(b,2,16,16)

        flow, correlation = local_correlation_softmax(feature0.contiguous(), feature1.contiguous(), local_radius=7, padding_mode='zeros')
        cor = self.conv1(correlation)
        #cor = grid*correlation
        cor = self.pool(cor)
        cor = self.conv2(cor)
        cor = self.pool(cor)#(b,16,8,8)
        grid_cor = coords_grid(b, 4, 4, device=cor.device)
        cor = bilinear_sample(cor, grid_cor, mode='bilinear', padding_mode='zeros')
        cor = self.conv3(cor)
        cor = self.pool(cor)



        #cor = bilinear_sample(cor, grid_cor, mode='bilinear', padding_mode='zeros')

        cor = cor.flatten(start_dim=1)
        max = torch.amax(torch.abs(flow), dim=(1, 2, 3))
        max = max.unsqueeze(-1) 

        cor = cor*(max-4)
        x = self.linear(cor)

        return x
    
import torch
import numpy as np
from thop import profile
# 确保你已经正确导入了你的 Router 类和相关依赖
# from your_module import Router 

def test_model_efficiency(model_path, img_size=(256, 256)):
    print(f"--- 开始测试模型, 输入分辨率: {img_size} ---")
    
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 实例化模型 (假设 opt 参数传 None)
    # 注意：这里需要确保你的 CNNEncoderRouter 等依赖都能正常工作
    model = Router(opt=None)

    # 3. 加载预训练权重
    try:
        print(f"正在加载权重文件: {model_path} ...")
        # map_location 确保在只有 CPU 的机器上也能加载 GPU 训练的模型
        state_dict = torch.load(model_path, map_location=device) 
        
        # 如果保存的是整个包含了优化器的 checkpoint，往往需要加上 ['state_dict'] 或 ['model']
        # if 'model' in state_dict:
        #     state_dict = state_dict['model']
            
        # 加载权重 (strict=False 可以在权重键名略有不匹配时防止直接报错)
        model.load_state_dict(state_dict, strict=True) 
        print("权重加载成功！")
    except Exception as e:
        print(f"加载权重失败，错误信息: {e}")
        print("请检查路径是否正确，或者 checkpoint 的键名是否匹配。")
        return

    # 将模型移动到对应设备并设置为评估模式
    model = model.to(device)
    model.eval()

    # 4. 构造虚拟输入
    h, w = img_size
    dummy_img0 = torch.randn(1, 3, h, w).to(device)
    dummy_img1 = torch.randn(1, 3, h, w).to(device)

    # ==========================================
    # 测试 1: 参数量 (Parameters) 和 计算量 (FLOPs)
    # ==========================================
    print("\n--- 开始计算 Parameters 和 FLOPs ---")
    try:
        # thop profile
        macs, params = profile(model, inputs=(dummy_img0, dummy_img1), verbose=False)
        macs_g = macs / 1e9
        params_m = params / 1e6
        print(f"Parameters (M): {params_m:.2f} M")
        print(f"FLOPs (G):      {macs_g:.2f} G (注: 此处为 MACs，部分论文视同 FLOPs)")
    except Exception as e:
        print(f"计算 FLOPs 失败 (可能是某些自定义算子不支持 thop): {e}")

    # ==========================================
    # 测试 2: 推理延迟 (Latency)
    # ==========================================
    print("\n--- 开始计算 Latency (GPU 毫秒级测试) ---")
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 100
        timings = np.zeros((repetitions, 1))

        print("正在进行 GPU 预热 (Warm-up)...")
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy_img0, dummy_img1)

        print(f"正在连续推理 {repetitions} 次以获取平均延迟...")
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_img0, dummy_img1)
                ender.record()
                torch.cuda.synchronize() # 强制等待 GPU 任务完成
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(f"Latency (ms):   {mean_syn:.2f} ms (+/- {std_syn:.2f} ms)")
    else:
        # 如果是 CPU 测试，可以直接用 time 模块
        import time
        print("注意: 正在 CPU 上测试延迟，结果通常远慢于真实 GPU 表现。")
        with torch.no_grad():
            for _ in range(10): # CPU 预热
                _ = model(dummy_img0, dummy_img1)
            
            start_time = time.time()
            for _ in range(10): 
                _ = model(dummy_img0, dummy_img1)
            end_time = time.time()
            avg_time_ms = (end_time - start_time) / 10 * 1000
            print(f"CPU Latency (ms): {avg_time_ms:.2f} ms")


if __name__ == "__main__":
    # 执行测试，请确保 router-306.pth 文件路径正确
    # 你可以修改 img_size 为你实际使用的分辨率，比如 (480, 640) 等
    test_model_efficiency(model_path="router_306.pth", img_size=(256, 256))



        


        
