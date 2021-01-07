from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class LoadData():
    def __init__(self, dataset = datasets.MNIST, batch_size = 64, transform = transforms.Compose([transforms.ToTensor(),])):  
        self.transform = transform
        self.batch_size = batch_size

        self.dataset = dataset

        self.train_data = dataset(
            root='../input/data',
            train=True,
            download=True,
            transform=transform
        )
        self.validate_data = dataset(
            root='../input/data',
            train=False,
            download=True,
            transform=transform
        )
        
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True
        )
        self.validate_loader = DataLoader(
            self.validate_data,
            batch_size=batch_size,
            shuffle=False
        )