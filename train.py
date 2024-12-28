import kagglehub
import torchvision
import torch
from get_dataloaders import get_dataloaders
from simple_diffusion import SimpleDiffusion
from model.unet import UNet
from engine import train
from inference import inference
from utils import load_config

config = load_config()

dataset_path = config["dataset_path"]
model_name = config["model_name"]
train_size = config["train_size"]
batch_size = config["batch_size"]
timesteps = config["timesteps"]
dropout = config["dropout"]
num_groups = config["num_groups"]
num_heads = config["num_heads"]
l_r = config["l_r"]
epoch = config["epochs"]
img_shape = (config["img_shape"][0], config["img_shape"][1], config["img_shape"][2])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = kagglehub.dataset_download(dataset_path)

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((img_shape[1], img_shape[2]),
                                      interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                      antialias=True),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Lambda(lambda t: (t*2)-1)
    ]
)

train_dataloader, test_dataloader = get_dataloaders(path, transforms, train_size, batch_size)

sd = SimpleDiffusion(timesteps=timesteps, img_shape=(batch_size, *img_shape), device=device)

model = UNet(dropout, num_groups, num_heads, timesteps).to(device)

optimizer = torch.optim.Adam(model.parameters(), l_r)

criterion = torch.nn.MSELoss()

train(epoch, model, optimizer, criterion, train_dataloader, test_dataloader, model_name, device, sd)

images = inference(model, sd, timesteps, (img_shape.shape[0],img_shape.shape[1],img_shape.shape[2]), 10, device)
