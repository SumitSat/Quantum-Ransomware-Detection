import os
import argparse
import time
from datetime import datetime
import torch
import torch.nn as nn
from src.dataset import get_dataloaders
from src.model import HybridQuantumNet
from src.model_qlstm import QTERD_QLSTM
from src.config import SEED, CHECKPOINT_DIR, N_CHANNELS

def train(device_string, epochs, batch_size, model_type="qlstm"):
    torch.manual_seed(SEED)
    
    print(f"Loading data for {model_type}...")
    train_loader, test_loader, pca_model, scaler_model = get_dataloaders(model_type=model_type)
    
    print(f"Initializing {model_type.upper()} Network on {device_string}...")
    if model_type == "cnn-vqc":
        model = HybridQuantumNet(device_string=device_string)
    else:
        model = QTERD_QLSTM(input_size=N_CHANNELS, hidden_size=16, n_qubits=4, n_vqc_layers=1, backend=device_string)
        
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_t = time.time()
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
        acc = correct / total
        elapsed = time.time() - start_t
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.4f} - Time: {elapsed:.2f}s")
        
    # Checkpoint Versioning
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    model_name = f"{model_type}_v1_{timestamp}.pt"
    model_path = os.path.join(CHECKPOINT_DIR, model_name)
    torch.save(model.state_dict(), model_path)
    
    # Save a symlink/copy as latest for easy loading
    latest_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_latest.pt")
    torch.save(model.state_dict(), latest_path)
    
    print(f"[{model_type.upper()}] Model Checkpoint saved to {model_path} and identically to {latest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="default.qubit", help="PennyLane device string")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--model", type=str, choices=["cnn-vqc", "qlstm"], default="qlstm", help="Model architecture")
    args = parser.parse_args()
    
    train(device_string=args.device, epochs=args.epochs, batch_size=args.batch_size, model_type=args.model)
