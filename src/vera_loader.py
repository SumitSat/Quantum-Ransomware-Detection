"""
VERA/Hybrid Analysis JSON Parser → QLSTM-Ready Tensors
---------------------------------------------------------
Parses the behavioral JSON files downloaded by fetch_hybrid_analysis.py
and converts them into fixed-length integer-encoded API sequences
suitable for the QLSTM model.

Pipeline:
  data/raw/hybrid_analysis/
      ransomware/*.json
      benign/*.json
      →  build_api_vocabulary()
      →  encode_sequences()
      →  data/processed/api_sequences.npz  (X, y, vocab)
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import Counter
from gensim.models import Word2Vec

# ─── Config ───────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw/hybrid_analysis")
PROCESSED_DIR = Path("data/processed")
VOCAB_PATH    = Path("checkpoints/api_vocab.json")
W2V_PATH      = Path("checkpoints/word2vec_api.model")

SEQ_LEN       = 120    # Fixed sequence length: pad/truncate to 120 API calls
EMBED_DIM     = 16     # Word2Vec embedding dimensions per API call
MIN_FREQ      = 3      # Minimum API call frequency to include in vocab
PAD_TOKEN     = "<PAD>"
UNK_TOKEN     = "<UNK>"


# ─── Step 1: Load raw JSON files ───────────────────────────────────────────────

def load_all_samples(raw_dir: Path = RAW_DIR) -> tuple[list, list]:
    """
    Load all JSON files from ransomware/ and benign/ subdirs.
    Returns:
        sequences : list of lists of API call name strings
        labels    : list of int (1=ransomware, 0=benign)
    """
    sequences, labels = [], []
    errors = 0

    for label_int, subdir in [(1, "ransomware"), (0, "benign")]:
        folder = raw_dir / subdir
        if not folder.exists():
            print(f"[WARN] Directory not found: {folder}")
            continue

        files = list(folder.glob("*.json"))
        print(f"Loading {len(files)} {subdir} samples...")

        for fpath in files:
            try:
                with open(fpath, "r") as f:
                    record = json.load(f)
                api_seq = record.get("api_sequence", [])
                if len(api_seq) >= 5:          # Skip near-empty sequences
                    sequences.append(api_seq)
                    labels.append(label_int)
            except Exception as e:
                errors += 1

    print(f"Loaded {len(sequences)} samples ({errors} errors skipped)")
    print(f"  Ransomware: {sum(labels)}  |  Benign: {len(labels) - sum(labels)}")
    return sequences, labels


# ─── Step 2: Build API vocabulary ─────────────────────────────────────────────

def build_api_vocabulary(sequences: list, min_freq: int = MIN_FREQ) -> dict:
    """
    Counts all API calls across all sequences.
    Assigns integer index to each API call that appears >= min_freq times.
    Returns vocab dict: {api_name: index}
    """
    counter = Counter()
    for seq in sequences:
        counter.update(seq)

    # Reserve 0=PAD, 1=UNK
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for api_name, count in counter.most_common():
        if count >= min_freq:
            vocab[api_name] = len(vocab)

    print(f"Vocabulary size: {len(vocab)} unique API calls (min_freq={min_freq})")
    return vocab


# ─── Step 3: Train Word2Vec embeddings ────────────────────────────────────────

def train_word2vec(sequences: list, embed_dim: int = EMBED_DIM) -> Word2Vec:
    """
    Trains Word2Vec on API call sequences treating them as 'sentences'.
    Each API call name becomes a dense embed_dim-dimensional vector.
    This captures semantic relationships between API calls
    (e.g., CryptAcquireContext and CryptCreateHash are embedding-neighbors).
    """
    print(f"Training Word2Vec embeddings (dim={embed_dim}) on {len(sequences)} sequences...")
    model = Word2Vec(
        sentences=sequences,
        vector_size=embed_dim,
        window=5,
        min_count=MIN_FREQ,
        workers=os.cpu_count(),
        epochs=10,
        seed=42
    )
    W2V_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(W2V_PATH))
    print(f"Word2Vec model saved to {W2V_PATH}")
    return model


# ─── Step 4: Encode sequences to fixed-length integer arrays ──────────────────

def encode_sequences(
    sequences: list,
    vocab: dict,
    seq_len: int = SEQ_LEN
) -> np.ndarray:
    """
    Converts each API call sequence (list of strings) to a fixed-length
    integer array of shape (seq_len,) using the vocabulary.

    Truncates sequences longer than seq_len (keeps LAST seq_len calls —
    this is intentional: the encryption-staging calls happen near the end).
    Pads shorter sequences at the START with PAD_TOKEN index (0).
    """
    X = np.zeros((len(sequences), seq_len), dtype=np.int32)
    unk_idx = vocab.get(UNK_TOKEN, 1)

    for i, seq in enumerate(sequences):
        # Keep last seq_len calls (pre-encryption activity is at the tail)
        truncated = seq[-seq_len:]
        # Pad at start
        start = seq_len - len(truncated)
        for j, api_name in enumerate(truncated):
            X[i, start + j] = vocab.get(api_name, unk_idx)

    return X


# ─── Step 5: Embed sequences using Word2Vec ───────────────────────────────────

def embed_sequences(
    X_int: np.ndarray,
    vocab: dict,
    w2v_model: Word2Vec,
    embed_dim: int = EMBED_DIM
) -> np.ndarray:
    """
    Converts integer-encoded sequences to dense float embeddings.
    Shape: (N_samples, SEQ_LEN, EMBED_DIM)

    This is the format fed directly to the QLSTM's Angle Encoding layer.
    """
    idx_to_api = {v: k for k, v in vocab.items()}
    N, L = X_int.shape
    X_emb = np.zeros((N, L, embed_dim), dtype=np.float32)

    wv = w2v_model.wv
    for i in range(N):
        for j in range(L):
            api_name = idx_to_api.get(X_int[i, j], PAD_TOKEN)
            if api_name in wv:
                X_emb[i, j] = wv[api_name]
            # else: leave as zeros (PAD or UNK with no W2V entry)

    return X_emb


# ─── Step 6: Save processed dataset ───────────────────────────────────────────

def save_processed(X_int: np.ndarray, y: np.ndarray, vocab: dict):
    """Save integer sequences, labels, and vocabulary to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outfile = PROCESSED_DIR / "api_sequences.npz"
    np.savez_compressed(outfile, X=X_int, y=y)
    print(f"Saved integer sequences to {outfile}  shape: X={X_int.shape}, y={y.shape}")

    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary ({len(vocab)} tokens) to {VOCAB_PATH}")


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def build_dataset(raw_dir: Path = RAW_DIR, use_word2vec: bool = True):
    """
    Full pipeline: load JSONs → vocab → encode → (optionally embed) → save.
    Returns X_int (N, SEQ_LEN), y (N,), vocab dict
    """
    sequences, labels = load_all_samples(raw_dir)
    if not sequences:
        raise RuntimeError(
            "No samples found! Run scripts/fetch_hybrid_analysis.py first."
        )

    vocab   = build_api_vocabulary(sequences)
    X_int   = encode_sequences(sequences, vocab)
    y       = np.array(labels, dtype=np.int32)

    save_processed(X_int, y, vocab)

    if use_word2vec:
        w2v = train_word2vec(sequences)
        X_emb = embed_sequences(X_int, vocab, w2v)
        emb_path = PROCESSED_DIR / "api_embeddings.npz"
        np.savez_compressed(emb_path, X=X_emb, y=y)
        print(f"Saved Word2Vec embeddings to {emb_path}  shape: X={X_emb.shape}")

    print("\n✅ Dataset build complete!")
    print(f"   Samples  : {len(y)}")
    print(f"   Sequence : {SEQ_LEN} timesteps per sample")
    print(f"   Vocab    : {len(vocab)} unique API calls")
    return X_int, y, vocab


if __name__ == "__main__":
    build_dataset()
