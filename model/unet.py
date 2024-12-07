import torch
from model.sinusoidal_embeddings import SinusoidalEmbeddings
from model.block import Block

class UNet(torch.nn.Module):
    def __init__(self, dropout, num_groups, num_heads, timesteps, channels = [3, 64, 128, 256, 512, 512, 384], attentions = [False, True, False, False, False, True]):
        super(UNet, self).__init__()
        self.resolutions_num = len(channels)
        self.first_conv = torch.nn.Conv2d(channels[0],channels[1], kernel_size = 3, padding = 1)
        channels_out = channels[-1]//2+channels[1]
        self.last_conv1 = torch.nn.Conv2d(channels_out,channels_out//2, kernel_size = 3, padding = 1)
        self.last_conv2 = torch.nn.Conv2d(channels_out//2,channels[0], kernel_size = 1)
        self.relu = torch.nn.ReLU(inplace = True)
        self.embedding = SinusoidalEmbeddings(timesteps=timesteps, embeddings_dim=max(channels))
        for i in range(1, self.resolutions_num):
            block = Block(channels = channels[i], dropout = dropout,
                          num_groups = num_groups, num_heads = num_heads,
                          attn = attentions[i-1], downsample = True if i-1<self.resolutions_num//2 else False)
            setattr(self, f'Block{i-1}', block)

    def forward(self, x, t):
        embeddings = self.embedding(x, t)
        x = self.first_conv(x)
        res_inputs = []
        for i in range(self.resolutions_num//2):
            layer = getattr(self, f'Block{i}')
            x, residual = layer(x, embeddings)
            res_inputs.append(residual)
        for i in range(self.resolutions_num//2, self.resolutions_num-1):
            layer = getattr(self, f'Block{i}')
            x = torch.concat((layer(x, embeddings)[0], res_inputs[self.resolutions_num-i-2]), dim=1)
        x = self.last_conv1(x)
        x = self.relu(x)
        x = self.last_conv2(x)
        return x
