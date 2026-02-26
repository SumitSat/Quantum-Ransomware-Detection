import os
import argparse
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.config import BASE_DIR, SEED, CHECKPOINT_DIR, N_CHANNELS
from src.dataset import get_dataloaders
from src.model import HybridQuantumNet
from src.model_qlstm import QTERD_QLSTM

def generate_shap_explanations(device_string="default.qubit", model_type="qlstm"):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"Loading test data for {model_type.upper()} SHAP analysis...")
    _, test_loader, _, _ = get_dataloaders(model_type=model_type)
    
    X_background = []
    X_test_sample = []
    
    for i, (X_batch, y_batch) in enumerate(test_loader):
        if i == 0:
            X_background.append(X_batch[:5])
            X_test_sample.append(X_batch[5:15])
        else:
            break
            
    X_background = torch.cat(X_background, dim=0)
    X_test_sample = torch.cat(X_test_sample, dim=0)
    
    print(f"Loading trained {model_type.upper()} on {device_string}...")
    if model_type == "cnn-vqc":
        model = HybridQuantumNet(device_string=device_string)
    else:
        model = QTERD_QLSTM(input_size=N_CHANNELS, hidden_size=16, n_qubits=4, n_vqc_layers=1, backend=device_string)
        
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_latest.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    
    model.eval()
    
    def model_wrapper(x_np):
        x_tensor = torch.FloatTensor(x_np)
        if len(x_tensor.shape) == 2 and model_type == "qlstm":
            # Reshape flat array back to sequence for KernelExplainer
            x_tensor = x_tensor.reshape(-1, 120, N_CHANNELS)
        with torch.no_grad():
            preds = model(x_tensor).numpy()
        return preds.reshape(-1, 1)

    print("Initializing KernelSHAP (Quantum-SHAP Approximation)...")
    
    # KernelExplainer requires 2D arrays, so we must flatten temporal inputs
    if model_type == "qlstm":
        bg_flat = X_background.reshape(X_background.shape[0], -1).numpy()
        test_flat = X_test_sample.reshape(X_test_sample.shape[0], -1).numpy()
    else:
        bg_flat = X_background.numpy()
        test_flat = X_test_sample.numpy()

    explainer = shap.KernelExplainer(model_wrapper, bg_flat)
    
    print("Calculating SHAP values (This will take a few minutes)...")
    shap_values = explainer.shap_values(test_flat)
    
    plots_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure()
    if model_type == "qlstm":
        # temporal_shap: shape [120]
        # We average absolute shap values over samples, then sum over features
        shap_values_3d = shap_values.reshape(X_test_sample.shape)
        temporal_impact = np.abs(shap_values_3d).mean(axis=0).sum(axis=1)
        
        plt.plot(range(120), temporal_impact, color='red', linewidth=2.5, label='QLSTM SHAP Impact')
        plt.axvspan(50, 90, color='yellow', alpha=0.3, label='30-70s Lead Time Zone')
        plt.xlabel("Time steps (0 to 120)")
        plt.ylabel("Mean SHAP Value (Impact on prediction)")
        plt.title("Quantum-Temporal SHAP Lead-Time Visualizer")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(plots_dir, f"{model_type}_temporal_shap.png")
    else:
        shap.summary_plot(shap_values, test_flat, feature_names=[f"PCA_{i}" for i in range(16)], show=False)
        plot_path = os.path.join(plots_dir, f"{model_type}_summary_plot.png")
        
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Quantum-SHAP explainability plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="default.qubit")
    parser.add_argument("--model", type=str, choices=["cnn-vqc", "qlstm"], default="qlstm")
    args = parser.parse_args()
    
    generate_shap_explanations(device_string=args.device, model_type=args.model)
