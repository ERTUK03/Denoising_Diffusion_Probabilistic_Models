import torch

class SimpleDiffusion:
    def __init__(self, timesteps, img_shape, device):
        self.timesteps = timesteps
        self.img_shape = img_shape
        self.device = device
        
        scale = 1000/self.timesteps
        self.beta = torch.linspace(scale*1e-4,0.02,self.timesteps,dtype=torch.float32,device=self.device)
        self.alpha = 1 - self.beta
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)

    def get_timesteps(self):
        return self.timesteps

    def get_alpha(self):
        return self.alpha

    def forward_diffusion(self, x0, timesteps):
        epsilon = torch.randn_like(x0)
        mean    = torch.sqrt(self.alpha_cumulative)[timesteps] * x0
        std_dev = torch.sqrt(1-self.alpha_cumulative[timesteps])
        sample  = mean + std_dev * epsilon
     
        return sample, epsilon
