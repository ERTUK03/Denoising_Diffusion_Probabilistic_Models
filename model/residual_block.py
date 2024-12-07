import torch

class Residual_Block(torch.nn.Module):
    def __init__(self, channels, num_groups, dropout):
        super(Residual_Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, padding = 1)
        self.gn1 = torch.nn.GroupNorm(num_groups = num_groups, num_channels = channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = dropout)
        self.conv2 = torch.nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, padding = 1)
        self.gn2 = torch.nn.GroupNorm(num_groups = num_groups, num_channels = channels)

    def forward(self, x, embeddings):
        x = x + embeddings[:,:x.shape[1],:,:]
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out+x
        return out
