import os
import argparse
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.dataset import get_dataloaders
from src.model import HybridQuantumNet
from src.model_qlstm import QTERD_QLSTM
from src.config import SEED, CHECKPOINT_DIR, N_CHANNELS

def evaluate(device_string, model_type="qlstm"):
    torch.manual_seed(SEED)
    
    print(f"Loading test data for {model_type}...")
    _, test_loader, _, _ = get_dataloaders(model_type=model_type)
    
    print(f"Initializing {model_type.upper()} Network on {device_string}...")
    if model_type == "cnn-vqc":
        model = HybridQuantumNet(device_string=device_string)
    else:
        model = QTERD_QLSTM(input_size=N_CHANNELS, hidden_size=16, n_qubits=4, n_vqc_layers=1, backend=device_string)
        
    latest_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_latest.pt")
    if os.path.exists(latest_path):
        model.load_state_dict(torch.load(latest_path, map_location='cpu'))
        print(f"Loaded checkpoint {latest_path}")
    else:
        print(f"Warning: No checkpoint found at {latest_path}. Using uninitialized weights.")
        
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            preds = (outputs >= 0.5).float()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n--- {model_type.upper()} Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="default.qubit", help="PennyLane device string")
    parser.add_argument("--model", type=str, choices=["cnn-vqc", "qlstm"], default="qlstm", help="Model architecture")
    args = parser.parse_args()
    
    evaluate(device_string=args.device, model_type=args.model)
