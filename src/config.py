import os
import torch
import pennylane as qml

# Random Seed for Reproducibility
SEED = 42

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Training Defaults
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# Model Architecture Config
N_CHANNELS = 10
SEQ_LEN = 120
N_QUBITS = 8
N_VQC_LAYERS = 2
PCA_COMPONENTS = 16
