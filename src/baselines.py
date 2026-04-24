import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from src.data_loader import get_dataloaders
from src.config import SEED

def evaluate_classical_baselines(fold=1):
    """
    Trains standard XGBoost and RandomForest classifiers on the exact
    same PyTorch Scaled Tensors as the Quantum models to ensure a fair
    mathematical benchmark for the JIST paper.
    """
    print("Loading VERA Testing Folds for Baselines...")
    dataloaders = get_dataloaders(n_folds=5, batch_size=1024) # We can load massive batches for classical
    
    # Grab the requested fold
    train_loader = dataloaders[fold - 1]['train_loader']
    test_loader = dataloaders[fold - 1]['test_loader']
    
    # Reconstruct the numpy matrices from the DataLoaders for sklearn/xgboost compat
    X_train, y_train = [], []
    for X_batch, y_batch in train_loader:
        X_train.append(X_batch.numpy())
        y_train.append(y_batch.numpy())
        
    X_test, y_test = [], []
    for X_batch, y_batch in test_loader:
        X_test.append(X_batch.numpy())
        y_test.append(y_batch.numpy())
        
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train).ravel()
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test).ravel()
    
    print(f"Classical Array Rebuilt: Train {X_train.shape}, Test {X_test.shape}")
    
    results = {}
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, eval_metric='logloss')
    }
    
    plt.figure(figsize=(8,6))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_scores)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC-ROC": auc
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
        
        print(f"--- {name} Metrics ---")
        print(f"Accuracy:  {acc:.4f} | F1 Score: {f1:.4f} | AUC-ROC: {auc:.4f}")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Classical Baselines (Fold {fold})')
    plt.legend(loc="lower right")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/classical_baselines_roc_fold{fold}.png")
    print("\nSaved overlayed baseline ROC curve to plots/classical_baselines_roc.png")
    
    return results

if __name__ == "__main__":
    evaluate_classical_baselines(fold=1)
