import torch
from get_dataloaders import MissingDataset
from load_datasets import load_datasets
from get_args import get_args


args = get_args()
save_path = './datasets/missing_datasets/ttt'+args.dataset_name+'_missing_'
X_miss_train, Z_miss_train, y_miss_train, mask_train, X_miss_val, Z_miss_val, y_miss_val, mask_val, X_miss_test, Z_miss_test, y_miss_test, mask_test = load_datasets(args)

train_dataset = MissingDataset(X_miss_train, Z_miss_train, y_miss_train, mask_train)
val_dataset = MissingDataset(X_miss_val, Z_miss_val, y_miss_val, mask_val)
test_dataset = MissingDataset(X_miss_test, Z_miss_test, y_miss_test, mask_test)

torch.save(train_dataset, save_path+'train.pth')
print("Train dataset saved successfully.")

# 검증 데이터셋 저장
torch.save(val_dataset, save_path+'val.pth')
print("Validation dataset saved successfully.")

# 테스트 데이터셋 저장
torch.save(test_dataset, save_path+'test.pth')
print("Test dataset saved successfully.")