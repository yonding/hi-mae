import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=10000)

# TRAIN AND VALIDATE
def train_and_validate(args, model, train_loader, val_loader, optimizer):
    
    patience = 10
    best_loss = 1e9
    best_epoch = 0
    counter = 0    

    for epoch in range(args.epochs):
        
        args.current_epoch = epoch

        model.train()
        train_loss, train_mse_loss, train_prediction_loss = train_one_epoch(args, model, train_loader, optimizer)

        model.eval()
        val_loss, val_mse_loss, val_prediction_loss = valid_model(args, model, val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_{args.model_name}.pth')
            counter = 0
        else:
            counter += 1

        if counter > patience:
            break

        print(f'Epoch: {epoch}')
        print(f'[TRAIN] Total loss: {train_loss:.4f}, MSE: {train_mse_loss:.4f}, Prediction Loss(CE or MSE): {train_prediction_loss:.4f}')
        if args.dataset_name != 'diabetes': # because diabetes is regression task
            train_metrics = calculate_metrics(args, model, train_loader)
            print(f'       Accuracy: {train_metrics["accuracy"]:.4f}, AUROC: {train_metrics["auroc"]:.4f}, F1 score: {train_metrics["f1_score"]:.4f}')
        print(f'[VALID] Total loss: {val_loss:.4f}, MSE: {val_mse_loss:.4f}, Prediction Loss(CE or MSE): {val_prediction_loss:.4f}')
        if args.dataset_name != 'diabetes': # because diabetes is regression task
            val_metrics = calculate_metrics(args, model, val_loader)
            print(f'       Accuracy: {val_metrics["accuracy"]:.4f}, AUROC: {val_metrics["auroc"]:.4f}, F1 score: {val_metrics["f1_score"]:.4f}')

    print(f'Best epoch: {best_epoch}, Best Val loss: {round(best_loss, 4)}')


# Calculate additional metrics: accuracy, AUROC, F1 score
def calculate_metrics(args, model, data_loader):
    y_true = []
    y_pred_prob = []

    model.eval()
    for batch_idx, (missing_data, complete_data, y, mask) in enumerate(data_loader):
        
        # Move to the CUDA device.
        missing_data = missing_data.to(args.device)
        complete_data = complete_data.to(args.device)
        y = y.to(args.device)
        mask = mask.to(args.device)

        y_pred, _ = model(missing_data, complete_data)
        y_pred_prob.extend(torch.softmax(y_pred, dim=1).tolist())

        y_true.extend(y.tolist())

    # Calculate ROC AUC with multi_class='ovr'
    auroc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')

    # Convert y_pred_prob to tensor
    y_pred_tensor = torch.tensor(y_pred_prob)

    # Calculate accuracy
    y_pred_labels = torch.argmax(y_pred_tensor, dim=1)
    accuracy = accuracy_score(y_true, y_pred_labels.tolist())

    # Calculate F1 score
    f1_score_val = f1_score(y_true, y_pred_labels.tolist(), average='macro')

    return {"accuracy": accuracy, "auroc": auroc, "f1_score": f1_score_val}


# TRAIN ONE EPOCHs
def train_one_epoch(args, model, train_loader, optimizer):

    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
    mse_loss = nn.MSELoss(reduction='none')
    mse_mean_loss = nn.MSELoss(reduction='mean')

    train_loss = 0
    train_prediction_loss = 0
    train_mse_loss = 0
    accuracy, Sensitivity, Specificity, Precision, f1 = 0, 0, 0, 0, 0

    model.train()
    for batch_idx, (missing_data, complete_data, y, mask) in enumerate(train_loader):
        
        # Move to the CUDA device.
        missing_data = missing_data.to(args.device)
        complete_data = complete_data.to(args.device)
        y = y.to(args.device)
        mask = mask.to(args.device)

        optimizer.zero_grad()

        y_pred, recon_data = model(missing_data, complete_data)

        loss = args.prediction_loss_rate * (cross_entropy_loss(y_pred, y.long()) if args.dataset_name != 'diabetes' else mse_mean_loss(y_pred, y.unsqueeze(1))) + args.mse_rate * torch.mean(torch.sum(mse_loss(recon_data, complete_data) * mask, dim=1))
        loss.backward()
        
        train_prediction_loss += cross_entropy_loss(y_pred, y.long()).item() if args.dataset_name != 'diabetes' else mse_mean_loss(y_pred, y.unsqueeze(1)).item()
        train_mse_loss += torch.mean(torch.sum(mse_loss(recon_data, complete_data) * mask, dim=1)).item()
        train_loss += loss.item()

        optimizer.step()

    return train_loss / len(train_loader), train_mse_loss / len(train_loader), train_prediction_loss / len(train_loader)


# VALIDATION
@torch.no_grad()
def valid_model(args, model, val_loader):
    cross_entropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss(reduction='none')
    mse_mean_loss = nn.MSELoss(reduction='mean')

    val_loss = 0
    val_mse_loss = 0
    val_prediction_loss = 0

    model.eval()
    for batch_idx, (missing_data, complete_data, y, mask) in enumerate(val_loader):
        
        # Move to the CUDA device.
        missing_data = missing_data.to(args.device)
        complete_data = complete_data.to(args.device)
        y = y.to(args.device)
        mask = mask.to(args.device)

        y_pred, recon_data = model(missing_data, complete_data)

        loss = args.prediction_loss_rate * (cross_entropy_loss(y_pred, y.long()) if args.dataset_name != 'diabetes' else mse_mean_loss(y_pred, y.unsqueeze(1))) + args.mse_rate * torch.mean(torch.sum(mse_loss(recon_data, complete_data) * mask, dim=1))

        val_mse_loss += torch.mean(torch.sum(mse_loss(recon_data, complete_data) * mask, dim=1)).item()
        val_prediction_loss += cross_entropy_loss(y_pred, y.long()).item() if args.dataset_name != 'diabetes' else mse_mean_loss(y_pred, y.unsqueeze(1)).item()
        val_loss += loss.item()

        # Print validation samples
        if args.current_epoch % args.print_period == 0 and batch_idx == 1:
            print(f"\nM: {missing_data[:1][:4]}")
            print(f"R: {recon_data[:1][:4]}")
            print(f"C: {complete_data[:1][:4]}")
    
    return val_loss / len(val_loader), val_mse_loss / len(val_loader), val_prediction_loss / len(val_loader)


# TEST
@torch.no_grad()
def test(args, model, test_loader):
    
    model.load_state_dict(torch.load(f'best_{args.model_name}.pth'))

    model.eval()
    test_loss, test_mse_loss, test_prediction_loss = valid_model(args, model, test_loader)
    
    print("\n========================= TEST RESULT =========================")
    print(f'[TEST] Total loss: {test_loss:.4f}, MSE: {test_mse_loss:.4f}, Prediction Loss(CE or MSE): {test_prediction_loss:.4f}')
    print("===============================================================")