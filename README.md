# Quantum-Temporal Early Ransomware Detection (Q-TERD)

## 📌 Project Overview
This project, **Q-TERD**, is a cutting-edge cybersecurity research initiative aimed at detecting ransomware in its pre-encryption staging phase. It leverages a hybrid **Quantum Machine Learning (QML)** architecture to provide Security Operations Center (SOC) analysts with an actionable "lead time" (30-70 seconds) to isolate infected nodes before any data is encrypted.

The research is developed to meet the standards of high-prestige journals such as **IEEE Transactions on Information Forensics and Security** and **Nature Communications**.

## 🎯 What We Are Trying to Achieve
Most ransomware detection systems focus on detecting anomalies *after* encryption begins, which is often too late. Our project's **Unique Selling Proposition (USP)** is preventing the encryption from occurring entirely. 

To achieve this, the project rests on 3 core pillars:
1. **Quantum-Temporal Modeling (QLSTM)**: Replacing standard 1D CNNs with a Hybrid Quantum-LSTM (QLSTM) to track long-term sequential API calls using parameterized Variational Quantum Circuits (VQCs).
2. **Research-Grade Data Synthesis**: Because real-world "pre-encryption" ransomware data is sparse, we synthetically stitch real ransomware API frequencies into benign execution traces using Poisson Point Processes and GMM background noise.
3. **Quantum-Inspired Feature Selection (QIEA)**: Implementing a Quantum-Inspired Evolutionary Algorithm (QIEA) to optimize and select the best API features without falling into local minima.

## 🚀 Current Progress Status
We have completed the structural codebase for the project. The Python architecture is fully modularized and ready for heavy compute.

- [x] **Phase 1: Grounding** - Setup of base `HybridQuantumNet` (VQC), `dataset.py`, `config.py`, and training/evaluation scripts.
- [x] **Phase 2: Realism** - Developed `src/data_loader.py` which dynamically synthesizes temporal sequences from static `CIC-MalDroid-2020` and `EMBER` datasets via Poisson generation and Contextual Injection.
- [x] **Phase 3: Novelty** - Programmed the advanced `QLSTMCell` architecture (`src/model_qlstm.py`) and the `QIEA` feature selector (`src/qiea.py`).
- [x] **Phase 4: Validation** - Integrated `KernelSHAP` in `src/explain.py` for generating Explainable AI (XAI) feature importance plots to validate the quantum models for journal reviewers.

## 🖥️ High-End College PC Execution Plan
Because Quantum simulations and deep learning models require heavy computational resources, the codebase has been prepared on a lightweight laptop but is designed to be executed on a High-End PC with a powerful GPU.

**Follow these exact steps on your High-End College PC:**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SumitSat/Quantum-Ransomware-Detection.git
   cd Quantum-Ransomware-Detection
   ```

2. **Install Dependencies**
   It's highly recommended to use a Conda environment or Python virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Provide the Raw Datasets**
   Ensure your datasets are placed exactly in the `data/raw/` directory.
   - `data/raw/feature_vectors_syscallsbinders_frequency_5_Cat.csv` *(CIC-MalDroid-2020)*
   - `data/raw/ember2018/train_features_0.jsonl` *(EMBER)*

4. **Run the Data Synthesis Pipeline**
   This script converts the static frequency data into mathematically rigorous temporal sequences.
   ```bash
   python src/data_loader.py
   ```
   *(This will create `data/synthetic/synthetic_ransomware_dataset.csv`)*

5. **Train the Model**
   Train the hybrid quantum model. We use `lightning.gpu` to utilize PennyLane's GPU-accelerated quantum simulator backend.
   ```bash
   python src/train.py --device lightning.gpu --epochs 10 --batch-size 128
   ```

6. **Evaluate the Model**
   Run the sliding-window evaluation script to calculate temporal accuracy.
   ```bash
   python src/evaluate.py --device lightning.gpu
   ```

7. **Generate XAI Validation Plots**
   Execute the Quantum-SHAP script to generate explainability plots for the research paper.
   ```bash
   python src/explain.py --device lightning.gpu
   ```

---
*Developed for target submission to JIST 2026 / IEEE TIFS.*
