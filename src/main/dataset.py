import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import sampler
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as dset
import torchvision.transforms as T

#Get chunker of the sampler
class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples



def get_loader(dataset_type, batch_size=1024, download=False):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    if dataset_type == 'cifar10':
        train_dataset = dset.CIFAR10('../datasets', train=True, download=download, transform=transform)
        test_dataset = dset.CIFAR10('../datasets', train=False, download=download, transform=transform)
    elif dataset_type == 'cifar100':
        train_dataset = dset.CIFAR100('../datasets', train=True, download=download, transform=transform)
        test_dataset = dset.CIFAR100('../datasets', train=False, download=download, transform=transform)
    elif dataset_type == 'flowers102':
        train_dataset = dset.Flowers102('../datasets', split='train', download=download, transform=transform)
        test_dataset = dset.Flowers102('../datasets', split='test', download=download, transform=transform)
    elif dataset_type == 'food101':
        train_dataset = dset.Food101('../datasets', split='train', download=download, transform=transform)
        test_dataset = dset.Food101('../datasets', split='test', download=download, transform=transform)
    else:
        raise ValueError("Unsupported dataset type")

    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

    # Define class labels if applicable
    classes = None
    if dataset_type == 'cifar10':
        classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    elif dataset_type == 'cifar100':
        classes = ()
    elif dataset_type == 'flowers102':
        classes = ()
    elif dataset_type == 'food101':
        classes = ()

    return trainloader, testloader, classes


def get_processed_loader(root='../datasets/ProcessedMiniImagenet', batch_size=32):

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Training set
    train_dataset = datasets.ImageFolder(os.path.join(root, 'Train'), transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test set
    test_dataset = datasets.ImageFolder(os.path.join(root, 'Test'), transform=transform)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    return trainloader, testloader, None


def get_dataset(id_dataset='cifar10', batch_size=1024, num_train=45000, num_val=5000, download=False):
    
    if (id_dataset=='cifar10'):

        print('Loading CIFAR 10')
        return get_loader(dataset_type=id_dataset, batch_size=batch_size, download=download)

    if (id_dataset=='cifar100'):

        print('Loading CIFAR 100')
        return get_loader(dataset_type=id_dataset,batch_size=batch_size, download=download)
    
    if (id_dataset=='miniimagenet'):

        print('Loading MINIIMAGENET')
        current_path = os.getcwd()
        return get_processed_loader(root=current_path+'/datasets/ProcessedMiniImagenet', batch_size=batch_size, num_train=num_train, num_val=num_val)
    
    if (id_dataset=='food101'):
        
        print('Loading FOOD101')
        return get_loader(dataset_type=id_dataset,batch_size=batch_size, download=download)
    
    if (id_dataset=='flowers102'):
        
        print('Loading FLOWERS102')
        return get_loader(dataset_type=id_dataset,batch_size=batch_size, download=download)
    
    