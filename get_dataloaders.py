import torch
from torch.utils.data import DataLoader, random_split
from get_dataset import get_dataset

def get_dataloaders(path, transforms, train_size, batch_size):
    dataset = get_dataset(path, transforms)
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, test_dataloader
