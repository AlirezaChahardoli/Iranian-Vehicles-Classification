"""
dataset.py

Data loading and preprocessing.
"""

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def get_dataloaders(dataset_path, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        transforms.Normalize((0.5,), (0.5,))

    ])



    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    labels = dataset.targets

    indices = list(range(len(dataset)))

    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels)


    train_data = Subset(dataset, train_idx)

    test_data = Subset(dataset, test_idx)


    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    return train_dataloader, test_dataloader, dataset.classes,  labels