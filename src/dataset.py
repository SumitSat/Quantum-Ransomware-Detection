import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.config import SYNTHETIC_DATA_DIR, SEED, BATCH_SIZE, PCA_COMPONENTS, N_CHANNELS, SEQ_LEN
from src.model import CNNEncoder

def load_raw_features():
    csv_path = os.path.join(SYNTHETIC_DATA_DIR, "synthetic_ransomware_dataset.csv")
    df = pd.read_csv(csv_path)
    
    # Exclude metadata columns
    feature_cols = [c for c in df.columns if c.startswith('seq_')]
    X_flat = df[feature_cols].values
    y = df['label'].values
    
    # Reshape to (N_SAMPLES, SEQ_LEN, N_CHANNELS)
    X = X_flat.reshape(-1, SEQ_LEN, N_CHANNELS)
    return X, y

def extract_cnn_features(X):
    # For Phase 1, we use an untrained initialized CNN as the encoder
    torch.manual_seed(SEED)
    cnn = CNNEncoder()
    cnn.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        embeddings = cnn(X_tensor).numpy()
    return embeddings

def get_dataloaders(test_size=0.2, model_type="cnn-vqc"):
    X, y = load_raw_features()
    
    if model_type == "cnn-vqc":
        # 1. Feature Extraction via CNN
        X_processed = extract_cnn_features(X)
        
        # 2. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=SEED, stratify=y
        )
        
        # 3. Scale and PCA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        pca = PCA(n_components=PCA_COMPONENTS)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
    else:
        # QLSTM Mode: Return pure sequences
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=SEED, stratify=y
        )
        # Scale each channel
        scaler = StandardScaler()
        N_tr, S, C = X_train.shape
        N_te, _, _ = X_test.shape
        X_train_flat = X_train.reshape(-1, C)
        X_test_flat = X_test.reshape(-1, C)
        
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_tr, S, C)
        X_test_scaled = scaler.transform(X_test_flat).reshape(N_te, S, C)
        
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
        pca = None
        
    # 4. Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_final), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_final), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, pca, scaler
