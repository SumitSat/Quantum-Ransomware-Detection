import os
import argparse
import time
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from src.data_loader import get_dataloaders
from src.model import HybridQuantumNet, StrictClassicalDNN
from src.config import SEED, CHECKPOINT_DIR

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train(device_string, epochs, batch_size, model_type="quantum"):
    torch.manual_seed(SEED)
    
    print("Loading VERA Static Dataset and preparing 5-Fold Stratified CV...")
    fold_dataloaders = get_dataloaders(n_folds=5, batch_size=batch_size)
    
    print(f"Initializing {model_type.upper()} Network on {device_string}...")
    
    fold_results = []
    
    for fold_data in fold_dataloaders:
        fold = fold_data['fold']
        train_loader = fold_data['train_loader']
        test_loader = fold_data['test_loader']
        input_dim = fold_data['input_dim']
        
        print(f"\n{'='*40}")
        print(f"Starting FOLD {fold}")
        print(f"{'='*40}")
        
        # Instantiate fresh model for each fold
        if model_type == "quantum":
            model = HybridQuantumNet(input_dim=input_dim, n_qubits=8, n_layers=2, device_string=device_string)
            model_name_prefix = "cnn_vqc"
        else:
            model = StrictClassicalDNN(input_dim=input_dim, hidden_dim=8)
            model_name_prefix = "dnn_classical"
            
        criterion = nn.BCELoss()
        # ML Robustness: L2 Regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4) # L2 penalty
        
        # ML Robustness: Early Stopping
        early_stopping = EarlyStopping(patience=3, verbose=True)
        
        # Store for patience recovery
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            start_t = time.time()
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                outputs = outputs.squeeze()
                y_batch = y_batch.squeeze()
                
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            elapsed = time.time() - start_t
            
            # Validation at end of epoch
            model.eval()
            all_preds = []
            all_labels = []
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch).squeeze()
                    y_batch_sq = y_batch.squeeze()
                    
                    batch_loss = criterion(outputs, y_batch_sq)
                    val_loss += batch_loss.item()
                    
                    preds = (outputs >= 0.5).float()
                    all_preds.extend(preds.numpy())
                    all_labels.extend(y_batch_sq.numpy())
                    
            val_acc = accuracy_score(all_labels, all_preds)
            avg_val_loss = val_loss / len(test_loader)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.2f}s")
            
            early_stopping(avg_val_loss)
            if early_stopping.counter == 0:  # If we hit a new best score
                best_model_state = model.state_dict()
                
            if early_stopping.early_stop:
                print("Early stopping triggered. Halting training for this fold to explicitly prevent overfitting.")
                break
                
        # Load the best model from the fold stopping criteria
        model.load_state_dict(best_model_state)
        fold_results.append(val_acc)
        
        # Save model per fold
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        model_name = f"{model_name_prefix}_fold{fold}_{timestamp}.pt"
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        model_path = os.path.join(CHECKPOINT_DIR, model_name)
        torch.save(model.state_dict(), model_path)
        
        latest_path = os.path.join(CHECKPOINT_DIR, f"{model_name_prefix}_fold{fold}_latest.pt")
        torch.save(model.state_dict(), latest_path)
        
    print("\n" + "="*40)
    print("5-FOLD CV COMPLETE")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1} Accuracy: {acc:.4f}")
    mean_acc = sum(fold_results) / len(fold_results)
    print(f"Mean Validation Accuracy: {mean_acc:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="default.qubit", help="PennyLane device string")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--model", type=str, choices=["quantum", "classical"], default="quantum", help="Which model to run")
    args = parser.parse_args()
    
    train(device_string=args.device, epochs=args.epochs, batch_size=args.batch_size, model_type=args.model)
