import os
import argparse
import time
import torch
import torch.nn as nn
from src.dataset import get_dataloaders
from src.model import HybridQuantumNet
from src.config import SEED, CHECKPOINT_DIR

def train(device_string, epochs, batch_size):
    torch.manual_seed(SEED)
    
    print(f"Loading data (PCA via CNN embeddings)...")
    train_loader, test_loader, pca_model, scaler_model = get_dataloaders()
    
    print(f"Initializing Hybrid Quantum Network on {device_string}...")
    model = HybridQuantumNet(device_string=device_string)
    
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
        
    model_path = os.path.join(CHECKPOINT_DIR, "vqc_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="default.qubit", help="PennyLane device string")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    train(device_string=args.device, epochs=args.epochs, batch_size=args.batch_size)
