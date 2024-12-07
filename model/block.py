import torch
from model.residual_block import Residual_Block
from model.attention_block import Attention_Block

class Block(torch.nn.Module):
    def __init__(self, channels, dropout, num_groups, num_heads, attn, downsample):
        super(Block, self).__init__()
        self.downsample = downsample
        self.res_block1 = Residual_Block(channels, num_groups, dropout)
        self.res_block2 = Residual_Block(channels, num_groups, dropout)
        if attn:
            self.attn = Attention_Block(channels, num_heads)
        if self.downsample:
            self.conv = torch.nn.Conv2d(in_channels = channels, out_channels = channels*2, kernel_size = 3, stride = 2, padding = 1)
        else:
            self.conv = torch.nn.ConvTranspose2d(in_channels = channels, out_channels = channels//2, kernel_size = 4, stride = 2, padding = 1)

    def forward(self, x, embeddings):
        x = self.res_block1(x, embeddings)
        if hasattr(self, 'attn'):
            x = self.attn(x)
        x = self.res_block2(x, embeddings)

        return self.conv(x), x
