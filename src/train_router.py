from core.dataset.dataloader import GenerateRouterDataloader
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import visdom
import random
import torch.utils.data 
import config_router
from core.router.router import Router
from core.trainer import trainer_router
torch.autograd.set_detect_anomaly(True)
opt = config_router.opt

def train(opt):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    viz = DrawVisdom()

    model = Router(opt).to(device)
    model.cuda()

    train_dataloader, valid_dataloader = GenerateRouterDataloader(opt.img_dir, opt.uv_dir, batch_size=opt.batch_size, scale=1, shift=False, pin_mem=False, training_size=opt.max_size)
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=1e-4)

    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)

    scheduler1 = StepLR(optimizer1, step_size=2, gamma=0.6)
    #scheduler2 = StepLR(optimizer2, step_size=20, gamma=0.5)

    train = trainer_router.Trainer(opt, model, optimizer1, 
                               scheduler1,train_dataloader, valid_dataloader , device, viz)
    
    train.train(opt.epochs)

class DrawVisdom():  
    def __init__(self, env_name='router_306', port=8097):  
        self.viz = visdom.Visdom(port=port, use_incoming_socket=False, env=env_name)  

  
    def plot_loss(self, epoch, loss, title='Training Loss'):  
        self.viz.line(Y=np.array([loss]), X=np.array([epoch]), win=title, update="append",
                 opts=dict(title=title, linecolor=np.array([[255, 140, 0]]), legend=[title]))
        
    def plot_points(self, points, title='Training Loss'):
        self.viz.scatter(
            X=points,
            win=title, 
            opts=dict(
                markersize=3,#markersize 参数设置了散点的大小为 10
                markercolor=np.array([[0, 140, 255]]),
                ),
            )
    
    def plot_image(self, image, title="Query image"):
        self.viz.images(
            image,
            win=title,
            opts=dict(title=title)
        )

    def plot_heatmap(self, image, title="uv"):
        self.viz.heatmap(image, win=title,
                         opts=dict(title=title))

def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything()
    train(opt)