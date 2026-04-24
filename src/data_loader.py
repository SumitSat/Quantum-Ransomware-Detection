"""
VERA Dataset Loader
-------------------
Extracts static PE features from the VERA dataset Zip file directly,
bypassing the need to unpack the 24GB zip.
It automatically handles string removal, NaN imputation, and StandardScaler.
"""

import os
import zipfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# The default path where the VERA zip will live on Vast.ai
VERA_ZIP_PATH = os.path.expanduser("~/vera_dataset/VERA_Dataset.zip")
CSV_INTERNAL_PATH = "VERA Dataset/Code/feature_dataset_true.csv"

def build_vera_tensors(zip_path=VERA_ZIP_PATH):
    """
    Reads the CSV directly out of the zip.
    Identifies the label column, drops non-numeric junk (filepaths, .text names),
    and returns PyTorch tensors (X, y).
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing VERA dataset: {zip_path}. Please download it first.")
        
    print(f"Opening {zip_path} and extracting static features. This may take 30-60 seconds...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(CSV_INTERNAL_PATH) as f:
            # We use low_memory=False because we have 550 columns of mixed types
            df = pd.read_csv(f, low_memory=False)
            
    print(f"Loaded CSV shape: {df.shape}")
    
    # 1. Identify the label
    # In VERA, the ransomware flag is often derived from the filepath or a specific label column.
    # Looking at the raw data, the filepath contains 'Ransomware' or 'Benign'.
    # We will search for a string column that looks like a path.
    str_cols = df.select_dtypes(include=['object']).columns
    
    path_col = None
    for c in str_cols:
        # Check if it looks like a path
        sample_val = str(df[c].iloc[0]).lower()
        if '/mnt/' in sample_val or '\\' in sample_val or 'dataset' in sample_val:
            path_col = c
            break
            
    if path_col is None:
        # Fallback: assume the last column is the label if no path is found
        labels = df.iloc[:, -1].astype(int).values
    else:
        # Generate binary labels from the filepath (1 = Ransomware, 0 = Benign)
        labels = df[path_col].apply(lambda x: 1 if 'ransomware' in str(x).lower() else 0).values

    # 2. Drop all string/object columns (like '.text', filepaths, hashes)
    df_numeric = df.select_dtypes(include=[np.number])
    
    # 2b. DATA LEAKAGE SCRUBBING: Drop structural/label columns mimicking features
    leakage_keywords = ['label', 'class', 'index', 'unnamed', 'id', 'score', 'family', 'hash']
    cols_to_drop = [c for c in df_numeric.columns if any(kw in c.lower() for kw in leakage_keywords)]
    
    if len(cols_to_drop) > 0:
        print(f"Scrubbing {len(cols_to_drop)} leaked columns to prevent artificial 99% accuracy: {cols_to_drop[:5]}...")
        df_numeric = df_numeric.drop(columns=cols_to_drop, errors='ignore')
    
    # 3. Handle NaNs
    df_numeric = df_numeric.fillna(0)
    
    # 4. Convert to Numpy
    X_raw = df_numeric.values
    y = np.array(labels, dtype=np.float32)
    
    print(f"Final Numeric Feature Matrix: {X_raw.shape}")
    print(f"Labels: {sum(y==1)} Ransomware | {sum(y==0)} Benign")
    
    return X_raw, y

def get_dataloaders(n_folds=5, batch_size=64):
    """
    Returns Stratified K-Fold DataLoaders for the VERA dataset.
    This fixes Fatal #3 and Critical #7 from the JIST audit.
    """
    X_raw, y = build_vera_tensors()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    dataloaders = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standard Scaling (Fit on Train, transform Train and Test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to Tensors
        tx_train = torch.tensor(X_train_scaled, dtype=torch.float32)
        ty_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        tx_test = torch.tensor(X_test_scaled, dtype=torch.float32)
        ty_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        
        train_ds = TensorDataset(tx_train, ty_train)
        test_ds = TensorDataset(tx_test, ty_test)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        dataloaders.append({
            'fold': fold + 1,
            'train_loader': train_loader,
            'test_loader': test_loader,
            'input_dim': X_train_scaled.shape[1] # To dynamically size the Model
        })
        
    return dataloaders
