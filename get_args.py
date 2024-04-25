import argparse
import torch

# PARSE ARGUMENTS
def get_args():
    parser = argparse.ArgumentParser(description="VAE for missing value imputation.")
    
    # DEVICE SETTINGS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device", default=device)

    # DATASET SETTINGS
    parser.add_argument("--DATASET_SETTINGS", default="----------------------")
    # wine, diabetes, boston, covtype, ortho
    # wine = 178 rows | 13 features (float) | 3 classes
    # diabetes = 442 rows | 10 features (float) | regression
    # boston = 506 rows | 13 features (float) | regression
    # covtype = 581012 rows | 54 features (int) | 7 classes
    parser.add_argument("--dataset_name", default="wine", type=str)
    parser.add_argument("--val_rate", default=0.2, type=float)
    parser.add_argument("--test", default=0.1, type=float)
    parser.add_argument("--subset_rate", default=0.0001, type=float)

    # GENERATION SETTINGS
    parser.add_argument("--GENERATION_SETTINGS", default="----------------------")
    # single, multiple, random
    parser.add_argument("--missing_pattern", default="multiple", type=str) 
    parser.add_argument("--include_complete", default=False, type=bool)
    # only used in SINGLE missing pattern
    parser.add_argument("--col_to_remove", default=2, type=int)
    # only used in MULTIPLE and RANDOM missing pattern
    parser.add_argument("--min_remove_count", default=1, type=int)       
    parser.add_argument("--max_remove_count", default=12, type=int)
    # only used in RANDOM missing pattern
    parser.add_argument("--new_num_per_origin", default=10, type=int)

    # MODEL SETTINGS
    parser.add_argument("--MODEL_SETTINGS", default="------------------------")
    parser.add_argument("--model_name",default="Tabular Transformer", type=str)
    parser.add_argument("--dim_model", default=256, type=int)
    parser.add_argument("--num_head", default=8, type=int)
    parser.add_argument("--dim_ff", default=256, type=int)
    # wine - 13/3 | diabetes - 10/reg | ortho - 21/2
    parser.add_argument("--num_features", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--threshold", default=0.5, type=int)
    parser.add_argument("--prediction_loss_rate", default=1, type=float)
    parser.add_argument("--mse_rate", default=1, type=float)
    parser.add_argument("--num_parameters", default=0, type=int)

    # LEARNING SETTINGS
    parser.add_argument("--LEARNING_SETTINGS", default="----------------------")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--step_size", default=20, type=int)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--print_period", default=10, type=int)

    return parser.parse_args()


def print_args(args):
    print("\n\n----------------- SETTINGS -------------------")
    
    not_to_print = {'single':['f', 'max_remove_count', 'new_num_per_origin'], 'multiple':['f', 'new_num_per_origin', 'col_to_remove'], 'random':['f', 'col_to_remove']}
    
    for arg, value in vars(args).items():
        if arg in ['DATASET_SETTINGS','GENERATION_SETTINGS', 'MODEL_SETTINGS', 'LEARNING_SETTINGS']:
            print("\n["+arg+"]")
            continue
        if arg not in not_to_print[args.missing_pattern]:
            print(f"{arg.ljust(max(len(arg) for arg in vars(args)))}: {value}")

    print("----------------------------------------------\n\n")

def set_nums(args):
    if args.dataset_name == 'wine':
        args.num_features = 13
        args.num_classes = 3
    elif args.dataset_name == 'diabetes':  
        args.num_features = 10
    elif args.dataset_name == 'ortho':  
        args.num_features = 21
        args.num_classes = 2