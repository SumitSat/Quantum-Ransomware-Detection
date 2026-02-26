import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.config import BASE_DIR, SEED, CHECKPOINT_DIR
from src.dataset import get_dataloaders
from src.model import HybridQuantumNet

def generate_shap_explanations(device_string="default.qubit"):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("Loading test data for SHAP analysis...")
    # Load reduced feature set
    _, test_loader, pca_model, scaler_model = get_dataloaders()
    
    # We only need a small background dataset for KernelExplainer
    X_background = []
    X_test_sample = []
    
    for i, (X_batch, y_batch) in enumerate(test_loader):
        # Taking 5 samples for background, 10 samples to explain to save compute time
        if i == 0:
            X_background.append(X_batch[:5]) 
            X_test_sample.append(X_batch[5:15]) 
        else:
            break
            
    X_background = torch.cat(X_background, dim=0)
    X_test_sample = torch.cat(X_test_sample, dim=0)
    
    print(f"Loading trained Q-TERD Hybrid Model on {device_string}...")
    model = HybridQuantumNet(device_string=device_string)
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, "vqc_model.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    else:
        print(f"Warning: No trained model checkpoint found at {ckpt_path}. Using uninitialized weights for dry-run.")
        
    model.eval()
    
    # Wrapper function for SHAP (expects numpy, returns numpy)
    def model_wrapper(x_np):
        x_tensor = torch.FloatTensor(x_np)
        with torch.no_grad():
            preds = model(x_tensor).numpy()
        return preds.reshape(-1, 1)

    print("Initializing KernelSHAP (Quantum-SHAP Approximation)...")
    explainer = shap.KernelExplainer(model_wrapper, X_background.numpy())
    
    print("Calculating SHAP values (This might take a minute)...")
    shap_values = explainer.shap_values(X_test_sample.numpy())
    
    print("Generating Summary Plot...")
    plt.figure()
    
    # SHAP summary plot
    shap.summary_plot(
        shap_values, 
        X_test_sample.numpy(), 
        feature_names=[f"PCA_{i}" for i in range(16)], 
        show=False
    )
    
    plots_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "qshap_summary_plot.png")
    
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Quantum-SHAP explainability plot saved to {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="default.qubit", help="PennyLane device string")
    args = parser.parse_args()
    
    generate_shap_explanations(args.device)
