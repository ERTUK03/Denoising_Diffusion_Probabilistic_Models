import torchvision

def get_dataset(path, transforms):
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transforms)
    return dataset
