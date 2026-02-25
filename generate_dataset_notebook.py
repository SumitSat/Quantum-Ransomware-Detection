
import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(content):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": content.splitlines(keepends=True)
    })

def add_code(content):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.splitlines(keepends=True)
    })

# Section 1
add_markdown("""# Synthetic Ransomware Dataset: Overview & Characterization
**Authors:** S.S.S., S.N.M. (Dr. Sumitra N. Motade)  
**Context:** Companion to "Hybrid Quantum-Classical Early Ransomware Detection"

## 1. Introduction
High-quality, labeled datasets that include explicit pre-encryption timestamps are scarce in public ransomware research due to privacy concerns and proprietary nature of telemetry. real endpoint telemetry is often missing precise annotations that mark the moment encryption begins.

To address this, we utilize a **Synthetic Time-Series Dataset** that simulates multivariate host telemetry (CPU usage, Memory I/O, Disk activity, Network counters) across a fixed observation window.

**Why Synthetic?**
1. **Ground Truth Precision**: We know exactly when encryption starts (`t_encrypt`) and when pre-attack drift begins (`t_detect`).
2. **Reproducibility**: Generated via fixed seeds, allowing consistent benchmarks.
3. **Control**: We can inject specific anomalous patterns (linear ramps, variance spikes) that mimic known ransomware behaviors like file enumeration and mass encryption.

This notebook visualizes the data structure, statistical properties, and characteristic signatures of benign vs. malicious samples.""")

# Section 2
add_markdown("""## 2. Dataset Generation Engine

The data is generated using a stochastic process:
- **Benign**: Gaussian noise with minor random spikes (transient system activity).
- **Malicious**: Base noise + Linear Drift (Preparation Phase) + High Amplitude Spike (Encryption Phase).

**Parameters:**
- **Samples**: 2,000
- **Sequence Length**: 120 timesteps
- **Channels**: 10 (simulating 10 different system metrics)
- **Class Balance**: 80% Benign, 20% Malicious
""")

add_code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization Settings
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# --- Generation Logic (Identical to Main Experiment) ---
def generate_dataset(n_samples=2000, seq_len=120, n_channels=10, mal_ratio=0.2):
    n_malicious = int(n_samples * mal_ratio)
    n_benign = n_samples - n_malicious
    
    X = np.zeros((n_samples, seq_len, n_channels))
    y = np.zeros(n_samples, dtype=int)
    
    # Metadata
    t_encrypt_gt = np.full(n_samples, np.nan)
    t_detect_gt = np.full(n_samples, np.nan) 
    
    # Random Seed for Consistency
    np.random.seed(42)
    
    # 1. Generate Benign (Noise + Transient Spikes)
    for i in range(n_benign):
        # Baseline noise
        X[i] = np.random.normal(0, 0.4393, (seq_len, n_channels)) + 0.0501 
        
        # Add random transient spikes (legitimate activity)
        num_spikes = np.random.randint(0, 3)
        for _ in range(num_spikes):
            spike_start = np.random.randint(0, seq_len-5)
            X[i, spike_start:spike_start+3, :] += np.random.normal(0.5, 0.2, (3, n_channels))
        
    # 2. Generate Malicious (Drift + Encryption Spike)
    for i in range(n_benign, n_samples):
        y[i] = 1
        
        # Determine Encryption Start (Late in sequence)
        start_t = np.random.randint(int(seq_len*0.6), seq_len-10)
        t_encrypt_gt[i] = start_t
        
        # Determine Early Warning Start (30-70 steps before encryption)
        lead_time = np.random.randint(30, 71) 
        drift_start = max(0, start_t - lead_time)
        t_detect_gt[i] = drift_start
        
        # Base Noise
        X[i] = np.random.normal(0, 0.6, (seq_len, n_channels))
        
        # Add Pre-Encryption Drift (Linear Ramp)
        # Simulates file enumeration / key generation / C2 comms
        drift_len = start_t - drift_start
        if drift_len > 0:
            drift = np.linspace(0, 1.0, drift_len)
            for c in range(n_channels):
                # Add drift to specific channels or all
                X[i, drift_start:start_t, c] += drift
        
        # Add Encryption Burst (High Amplitude)
        # Simulates mass file I/O and CPU pinning
        enc_len = seq_len - start_t
        X[i, start_t:, :] += np.random.normal(2.0, 1.5, (enc_len, n_channels))

    return X, y, t_encrypt_gt, t_detect_gt

# Generate Data
X_raw, y, t_enc, t_det = generate_dataset()
print("Dataset Generated Successfully.")
print(f"Dimensions: {X_raw.shape} (Samples, Timesteps, Channels)")""")

# Section 3
add_markdown("""## 3. Statistical Profile

A quick look at the global statistics separates the valid "normal" range from the "anomalous" range found in malicious samples.
""")

add_code("""# Create DataFrame for Global Stats
flat_data = X_raw.reshape(-1, 10)
columns = [f"Ch_{i}" for i in range(10)]
df_flat = pd.DataFrame(flat_data, columns=columns)
df_flat['Label'] = np.repeat(y, 120)

print("--- Global Statistics ---")
print(df_flat.describe().T[['mean', 'std', 'min', 'max']])

# Class Balance
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="viridis")
plt.title("Class Distribution (0=Benign, 1=Malicious)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0, 1], ['Benign (1600)', 'Malicious (400)'])
plt.show()""")

# Section 4
add_markdown("""## 4. Visualizing Ransomware Signatures

This is the most critical visualization. We compare a typical Benign trajectory with a Malicious one to highlight the **"Window of Opportunity"**.

- **Green Dotted Line**: `t_detect_gt`. The moment varying behavior begins (e.g., file enumeration). This is where we WANT to detect.
- **Red Dashed Line**: `t_encrypt_gt`. The moment mass encryption starts. This is where damage becomes irreversible.
""")

add_code("""# Select specific samples for clarity
benign_idx = 0 
mal_idx = 1605 # Arbitrary malicious sample index

# Plot Comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Benign Plot
axes[0].plot(X_raw[benign_idx], alpha=0.6)
axes[0].set_title("Benign Sample (Normal Activity)")
axes[0].set_xlabel("Time (steps)")
axes[0].set_ylabel("Metric Amplitude (Normalized)")
axes[0].grid(True)

# Malicious Plot
axes[1].plot(X_raw[mal_idx], alpha=0.6)
# Annotate Ground Truths
enc_t = t_enc[mal_idx]
det_t = t_det[mal_idx]

axes[1].axvline(det_t, color='green', linestyle=':', linewidth=3, label='Early Warning Start (Drift)')
axes[1].axvline(enc_t, color='red', linestyle='--', linewidth=3, label='Encryption Start')
axes[1].axvspan(det_t, enc_t, color='yellow', alpha=0.1, label='Detection Window')
axes[1].axvspan(enc_t, 120, color='red', alpha=0.1, label='Damage Phase')

axes[1].set_title(f"Malicious Sample (Ransomware Attack) \n Lead Time: {enc_t - det_t} steps")
axes[1].set_xlabel("Time (steps)")
axes[1].legend(loc='upper left')
axes[1].grid(True)

plt.tight_layout()
plt.show()""")

add_markdown("""### Heatmap View
Visualizing all 10 channels simultaneously for the malicious sample shows the correlation of the attack across system metrics (e.g., CPU and Disk spiking together).
""")

add_code("""plt.figure(figsize=(14, 5))
sns.heatmap(X_raw[mal_idx].T, cmap="coolwarm", center=0)
plt.title("Multi-Channel Heatmap: Malicious Sample Activity")
plt.xlabel("Time (steps)")
plt.ylabel("Channel (Metric)")
# Draw event lines
plt.axvline(det_t, color='green', linestyle=':', linewidth=2)
plt.axvline(enc_t, color='red', linestyle='--', linewidth=2)
plt.show()""")

# Section 5
add_markdown("""## 5. Temporal Distributions

Understanding the distribution of `t_encrypt` is vital. If encryption happens too early (e.g., step 5), there is no data to detect. Our dataset places encryption late (60-90% of window) to allow for sufficient history.
""")

add_code("""# Filter out NaNs (Benign samples have no encryption time)
valid_t_enc = t_enc[~np.isnan(t_enc)]
valid_t_det = t_det[~np.isnan(t_det)]
lead_times = valid_t_enc - valid_t_det

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

# Encryption Start Distribution
sns.histplot(valid_t_enc, kde=True, ax=ax[0], color='red', bins=20)
ax[0].set_title("Distribution of Encryption Start Times (t_encrypt)")
ax[0].set_xlabel("Timestep")

# Lead Time Distribution
sns.histplot(lead_times, kde=True, ax=ax[1], color='orange', bins=15)
ax[1].set_title("Available Lead Time (Window of Opportunity)")
ax[1].set_xlabel("Steps (t_encrypt - t_detect)")

plt.show()""")

# Section 6
add_markdown("""## 6. Data Export Structure

For compatibility with classical ML libraries (Scikit-Learn, XGBoost), the 3D data `(2000, 120, 10)` is often flattened into 2D `(2000, 1200)`.

**CSV Layout:**
- `seq_0_0` ... `seq_119_9`: Feature columns.
- `label`: 0 or 1.
- `t_encrypt`: Ground truth integer.
- `t_detect`: Ground truth integer.
""")

add_code("""# Create Flattened DataFrame representation
# Reshape (N, 120, 10) -> (N, 1200)
X_flat = X_raw.reshape(X_raw.shape[0], -1)
col_names = [f"seq_{t}_{c}" for t in range(120) for c in range(10)]

df_export = pd.DataFrame(X_flat, columns=col_names)
df_export['label'] = y
df_export['t_encrypt_gt'] = t_enc
df_export['t_detect_gt'] = t_det

print(f"Export DataFrame Shape: {df_export.shape}")
print("\\nFirst 5 rows (Preview):")
display(df_export.head())

# Optional: Save to CSV
# df_export.to_csv("synthetic_ransomware_dataset.csv", index=False)""")

filename = os.path.join(r"c:\\Users\\Shashwat\\OneDrive\\Desktop\\ransomeware", "Dataset_Preview_and_Overview.ipynb")
with open(filename, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook generated successfully at: {filename}")
