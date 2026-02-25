import os
import torch
import numpy as np
import pandas as pd
from src.dataset import get_dataloaders
from src.model import HybridQuantumNet
from src.config import SEED, CHECKPOINT_DIR

def evaluate(device_string):
    torch.manual_seed(SEED)
    print("Loading data...")
    train_loader, test_loader, pca_model, scaler_model = get_dataloaders()
    
    model_path = os.path.join(CHECKPOINT_DIR, "vqc_model.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return
        
    print(f"Loading trained Quantum Network on {device_string}...")
    model = HybridQuantumNet(device_string=device_string)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Squeeze to handle potential 1D dimension matching
            outputs = model(X_batch)
            if len(outputs.shape) > 1:
                outputs = outputs.squeeze(1)
            
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
    print(f"Test Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="default.qubit", help="PennyLane device string")
    args = parser.parse_args()
    
    evaluate(args.device)
