import torch
import torch.optim as optim
from get_args import get_args, print_args, set_nums
from get_dataloaders import get_train_dataloader, get_val_dataloader, get_test_dataloader, print_dataloaders_shape
from train_valid_test import train_and_validate, test
from tabular_transformer import TabularTransformer

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=10000)

def main(args):

    # 1. Define the model and optimizer
    model = TabularTransformer(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    args.num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_args(args)

    # 2. Load the dataset
    train_dataset = torch.load(f'./datasets/missing_datasets/{args.dataset_name}_missing_train.pth')
    val_dataset = torch.load(f'./datasets/missing_datasets/{args.dataset_name}_missing_val.pth')
    test_dataset = torch.load(f'./datasets/missing_datasets/{args.dataset_name}_missing_test.pth')
    
    # 2. Train and validate the model
    train_and_validate(args, model, train_dataset, val_dataset, optimizer)

    # 3. Test the model
    test(args, model, test_dataset)

if __name__ == "__main__":
    args = get_args()
    set_nums(args)
    main(args)
    print_args(args)
