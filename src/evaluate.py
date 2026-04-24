import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from src.data_loader import get_dataloaders
from src.model import HybridQuantumNet
from src.config import SEED, CHECKPOINT_DIR

def evaluate(device_string, fold=1):
    torch.manual_seed(SEED)
    
    print("Loading VERA Test Data...")
    dataloaders = get_dataloaders(n_folds=5, batch_size=128)
    
    # Grab the requested fold
    test_loader = dataloaders[fold - 1]['test_loader']
    input_dim = dataloaders[fold - 1]['input_dim']
    
    print(f"Initializing Hybrid Quantum CNN-VQC Network on {device_string}...")
    model = HybridQuantumNet(input_dim=input_dim, n_qubits=8, n_layers=2, device_string=device_string)
        
    latest_path = os.path.join(CHECKPOINT_DIR, f"cnn_vqc_fold{fold}_latest.pt")
    # Due to naming we might just try to load the most recent file if this fails
    if os.path.exists(latest_path):
        model.load_state_dict(torch.load(latest_path, map_location='cpu'))
        print(f"Loaded checkpoint {latest_path}")
    else:
        print(f"Warning: No checkpoint found at {latest_path}. Make sure you trained first.")
        
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            preds = (outputs >= 0.5).float()
            
            y_true.extend(y_batch.squeeze().numpy())
            y_pred.extend(preds.numpy())
            y_scores.extend(outputs.numpy())
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_scores)
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - VERA Quantum VQC')
        plt.legend(loc="lower right")
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/roc_curve.png")
        print("\nROC Curve saved to plots/roc_curve.png")
    except Exception as e:
        auc = 0.0
        print(f"Could not compute AUC: {e}")
    
    print(f"\n--- CNN-VQC Evaluation Metrics (Fold {fold}) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("---------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="default.qubit", help="PennyLane device string")
    parser.add_argument("--fold", type=int, default=1, help="Which fold to evaluate")
    args = parser.parse_args()
    
    evaluate(device_string=args.device, fold=args.fold)
