import torch
import torch.nn as nn
import os

def searchForMaxIteration(folder):
    # 这个好理解一眼就是在找对应路径下最大迭代次数对应的文件夹在哪
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

class AppModel(nn.Module):
    def __init__(self, num_images=1600):  
        super().__init__()   
        self.appear_ab = nn.Parameter(torch.zeros(num_images, 2).cuda()) # 每个图像都有两个补偿参数ab
        self.optimizer = torch.optim.Adam([
                                {'params': self.appear_ab, 'lr': 0.001, "name": "appear_ab"},
                                ], betas=(0.9, 0.99)) # 分别设置对应的学习率
            
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "app_model/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        print(f"save app model. path: {out_weights_path}")
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'app.pth')) # 曝光补偿模型的相关参数保存

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "app_model"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "app_model/iteration_{}/app.pth".format(loaded_iter))
        state_dict = torch.load(weights_path)
        print(f"load app model {weights_path}")
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)
