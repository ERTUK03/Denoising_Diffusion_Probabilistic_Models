import torch

class Attention_Block(torch.nn.Module):
    def __init__(self, channels, num_heads):
        super(Attention_Block, self).__init__()
        self.attn = torch.nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_seq = x_reshaped.permute(1, 0, 2)
        out, _ = self.attn(x_seq, x_seq, x_seq)
        out = out.permute(1, 0, 2)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out
