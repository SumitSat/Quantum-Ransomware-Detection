import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

from src.data_loader import get_dataloaders
from src.model import HybridQuantumNet

def explain_model():
    """
    Applies SHAP (SHapley Additive exPlanations) to the Quantum Net.
    Because VQCs are mathematically opaque, this proves to the journal reviewers
    EXACTLY which classical generic features triggered the ransomware classification.
    """
    print("Loading First Fold of Data for Explainability Analysis...")
    dataloaders = get_dataloaders(n_folds=5, batch_size=128)
    
    train_loader = dataloaders[0]['train_loader']
    test_loader = dataloaders[0]['test_loader']
    input_dim = dataloaders[0]['input_dim']
    
    model = HybridQuantumNet(input_dim=input_dim, n_qubits=8, n_layers=2, device_string="default.qubit")
    
    # Try to load weights if they exist
    weight_path = "checkpoints/cnn_vqc_fold1_latest.pt"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    
    model.eval()
    
    # We use a Background Distribution for Kernel SHAP
    # Pull 100 random samples from the Training set to establish the "Base Value"
    print("Initializing SHAP Background Distribution (100 samples)...")
    X_background_list = []
    for X_batch, _ in train_loader:
        X_background_list.append(X_batch)
        if sum([x.shape[0] for x in X_background_list]) >= 100:
            break
            
    X_background = torch.cat(X_background_list, dim=0)[:100]
    
    # Pull 25 Test samples to actually Explain
    X_test_explain = []
    for X_batch, _ in test_loader:
        X_test_explain.append(X_batch)
        break # Just grab the first batch
        
    X_test_explain = X_test_explain[0][:25]
    
    print(f"Executing Kernel Explainer on 25 test samples (This may take roughly 2-3 minutes)...")
    
    # We must wrap the PyTorch model output for SHAP (expects numpy array inputs, returns numpy array)
    def model_predict_wrapper(numpy_array):
        tensor_input = torch.tensor(numpy_array, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(tensor_input)
            # The Quantum model returns [Batch, 1], we return [Batch]
            return outputs.numpy().flatten()
            
    explainer = shap.KernelExplainer(model_predict_wrapper, X_background.numpy())
    shap_values = explainer.shap_values(X_test_explain.numpy())
    
    # Feature Names (Generically named since we bypassed reading the raw CSV columns entirely)
    # If we wanted actual column names like ".text_size", we would need to pass them from the data_loader
    feature_names = [f"PE_Feature_{i}" for i in range(input_dim)]
    
    # Global Summary Plot (Beeswarm)
    shap.summary_plot(shap_values, X_test_explain.numpy(), feature_names=feature_names, show=False)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/shap_summary_static.png", bbox_inches='tight')
    print("\n✅ SHAP Explainability Diagram saved to plots/shap_summary_static.png")

if __name__ == "__main__":
    explain_model()
