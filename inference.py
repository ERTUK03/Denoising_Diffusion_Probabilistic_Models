import torch

def inference(model, sd, timesteps, img_shape, img_num, device):
    x = torch.randn((img_num, *img_shape), device = device)
    for i in reversed(range(1, timesteps)):
        z = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
        ts = torch.ones(img_num, dtype=torch.long, device = device) * i
        pred_noise = model(x, ts)
        alpha = sd.get_alpha()[i]
        x = (1/torch.sqrt(alpha))*(x - ((1-alpha)/torch.sqrt(1-alpha))*pred_noise)+ torch.sqrt(1-alpha)*z
    return x
