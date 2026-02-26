import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from imblearn.combine import SMOTETomek
from src.config import RAW_DATA_DIR, SYNTHETIC_DATA_DIR, SEQ_LEN, N_CHANNELS, SEED

np.random.seed(SEED)

def _extract_maldroid_frequencies(num_samples=500):
    mal_path = os.path.join(RAW_DATA_DIR, "feature_vectors_syscallsbinders_frequency_5_Cat.csv")
    if not os.path.exists(mal_path):
        return None
    df = pd.read_csv(mal_path)
    features = df.drop(columns=['Class'], errors='ignore').values[:num_samples]
    return features

def _extract_ember_frequencies(num_samples=500):
    ember_path = os.path.join(RAW_DATA_DIR, "ember2018", "train_features_0.jsonl")
    if not os.path.exists(ember_path):
        return None
    
    features = []
    with open(ember_path, 'r') as f:
        for _ in range(num_samples):
            try:
                line = f.readline()
                if not line: break
                data = json.loads(line)
                features.append(data.get("histogram", [0]*256))
            except json.JSONDecodeError:
                break
    return np.array(features)

def generate_poisson_sequences(rates, seq_len):
    n_samples, n_channels = rates.shape
    lambdas = np.clip(rates / seq_len, a_min=0.01, a_max=None)
    sequences = np.zeros((n_samples, seq_len, n_channels))
    for i in range(n_samples):
        for c in range(n_channels):
            sequences[i, :, c] = np.random.poisson(lambdas[i, c], seq_len)
    return sequences

def contextual_injection(benign_seq, mal_seq):
    n_samples, seq_len, n_channels = benign_seq.shape
    injected_seq = np.copy(benign_seq)
    
    for i in range(n_samples):
        t_start = np.random.randint(10, seq_len // 2)
        duration = np.random.randint(10, seq_len // 3)
        t_end = min(t_start + duration, seq_len)
        alpha = np.random.uniform(0.6, 0.9) 
        mal_component = mal_seq[i % len(mal_seq), 0:(t_end-t_start), :]
        if mal_component.shape[0] < (t_end - t_start):
            continue
            
        injected_seq[i, t_start:t_end, :] = (alpha * mal_component + 
                                             (1-alpha) * benign_seq[i, t_start:t_end, :])
    return injected_seq

def apply_gmm_noise(sequences):
    shape = sequences.shape
    flat_seq = sequences.reshape(-1, shape[-1])
    sample_size = min(1000, len(flat_seq))
    gmm = GaussianMixture(n_components=2, random_state=SEED)
    gmm.fit(flat_seq[:sample_size])
    noise, _ = gmm.sample(shape[0] * shape[1])
    noise = noise.reshape(shape)
    noisy_seq = sequences + (0.1 * noise)
    noisy_seq = np.clip(noisy_seq, 0, None)
    return noisy_seq

def build_dataset():
    print("1. Extracting MalDroid and EMBER frequencies...")
    mal_freq = _extract_maldroid_frequencies()
    benign_freq = _extract_ember_frequencies()
    
    if mal_freq is None or len(mal_freq) == 0 or benign_freq is None or len(benign_freq) == 0:
        print("Missing dataset files. Proceeding with dummy random generation...")
        mal_freq = np.random.rand(500, 100) * 100
        benign_freq = np.random.rand(500, 100) * 50
        
    print("2. Projecting down to configured N_CHANNELS via PCA...")
    pca_mal = PCA(n_components=N_CHANNELS)
    mal_reduced = np.abs(pca_mal.fit_transform(mal_freq))
    
    pca_ben = PCA(n_components=N_CHANNELS)
    ben_reduced = np.abs(pca_ben.fit_transform(benign_freq))
    
    print("3. Synthesizing temporal traces using Poisson Point Processes...")
    mal_seqs_raw = generate_poisson_sequences(mal_reduced, SEQ_LEN)
    benign_seqs = generate_poisson_sequences(ben_reduced, SEQ_LEN)
    
    print("4. Applying Contextual Injection and GMM Noise...")
    ransomware_seqs = contextual_injection(benign_seqs, mal_seqs_raw)
    benign_seqs = apply_gmm_noise(benign_seqs)
    ransomware_seqs = apply_gmm_noise(ransomware_seqs)
    
    X = np.concatenate([benign_seqs, ransomware_seqs], axis=0)
    y = np.concatenate([np.zeros(len(benign_seqs)), np.ones(len(ransomware_seqs))])
    
    print("5. Balancing with SMOTE-Tomek...")
    smote = SMOTETomek(random_state=SEED)
    X_flat = X.reshape(X.shape[0], -1)
    X_res, y_res = smote.fit_resample(X_flat, y)
    
    cols = [f"seq_{i}" for i in range(X_res.shape[1])]
    df = pd.DataFrame(X_res, columns=cols)
    df['label'] = y_res
    
    out_path = os.path.join(SYNTHETIC_DATA_DIR, "synthetic_ransomware_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Research-Grade Synthetic Dataset built and saved to {out_path}!")

if __name__ == "__main__":
    build_dataset()
