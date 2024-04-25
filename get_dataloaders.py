import torch
from load_datasets import load_datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def get_train_dataloader(args):
    train_dataset = torch.load(f'./datasets/missing_datasets/{args.dataset_name}_missing_train.pth')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_loader

def get_val_dataloader(args):
    val_dataset = torch.load(f'./datasets/missing_datasets/{args.dataset_name}_missing_val.pth')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return val_loader

def get_test_dataloader(args):
    test_dataset = torch.load(f'./datasets/missing_datasets/{args.dataset_name}_missing_test.pth')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return test_loader

class MissingDataset(Dataset):
    def __init__(self, missing_data, complete_data, y, mask):
        # subset_indices = torch.randperm(len(missing_data))[:int(0.00001*len(missing_data))]
        # self.missing_data = missing_data[subset_indices]
        # self.complete_data = complete_data[subset_indices]
        # self.y = y[subset_indices]
        # self.mask = mask[subset_indices]
        self.missing_data = missing_data
        self.complete_data = complete_data
        self.y = y
        self.mask = mask

    def __len__(self):
        return len(self.missing_data)
    
    def __getitem__(self, idx):
        return self.missing_data[idx], self.complete_data[idx], self.y[idx], self.mask[idx]
    
    def sampling(self, sampling_rate):
        subset_indices = torch.randperm(len(self.missing_data))[:int(sampling_rate*len(self.missing_data))]
        self.missing_data = self.missing_data[subset_indices]
        self.complete_data = self.complete_data[subset_indices]
        self.y = self.y[subset_indices]
        self.mask = self.mask[subset_indices]

def print_dataloaders_shape(args, train_dataset, val_dataset):
    print("\n----------------- DATA SHAPE -----------------")
    print(f"TRAIN  : ({int(train_dataset.complete_data.shape[0]*args.subset_rate)}, {train_dataset.complete_data.shape[1]})")
    print(f"VALID  : ({int(val_dataset.complete_data.shape[0]*args.subset_rate)}, {val_dataset.complete_data.shape[1]})")
    # print(f"TEST   : ({len(test_loader.dataset)//args.batch_size * args.batch_size}, {next(iter(test_loader))[0].shape[1]})")
    print("----------------------------------------------\n")

def print_testloader_shape(args, test_dataset):
    print("\n----------------- DATA SHAPE -----------------")
    print(f"TEST   : ({int(test_dataset.complete_data.shape[0]*args.subset_rate)}, {test_dataset.complete_data.shape[1]})")
    print("----------------------------------------------\n")

def deep_copy_dataset(dataset):
    return MissingDataset(dataset.missing_data.clone(), dataset.complete_data.clone(), dataset.y.clone(), dataset.mask.clone())