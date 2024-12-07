import torch

class SinusoidalEmbeddings(torch.nn.Module):
    def __init__(self, timesteps, embeddings_dim):
        super(SinusoidalEmbeddings, self).__init__()
        factor = 10000**(torch.arange(0,embeddings_dim//2).float()/(embeddings_dim//2))
        
        self.embeddings = (torch.arange(timesteps, requires_grad=False).view(timesteps, 1).repeat(1, embeddings_dim//2)/factor).repeat_interleave(2, dim=1)
        self.embeddings[:,0::2] = torch.sin(self.embeddings[:,0::2])
        self.embeddings[:,1::2] = torch.cos(self.embeddings[:,1::2])
        
    def forward(self, x, timestep):
        return self.embeddings.to(x.device)[timestep][:, :, None, None]
