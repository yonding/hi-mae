import torch
import torch.optim as optim
from get_args import get_args, print_args
from get_dataloaders import get_dataloaders
from train_valid_test import train_and_validate, test
from tabular_transformer import TabularTransformer

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=10000)

def main(args):

    # 1. Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(args)
    
    # 2. Define the model and optimizer
    model = TabularTransformer(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    args.num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_args(args)

    # 3. Train and validate the model
    train_and_validate(args, model, train_loader, val_loader, optimizer)

    # 4. Test the model
    test(args, model, test_loader)

if __name__ == "__main__":
    args = get_args()
    main(args)
    print_args(args)
