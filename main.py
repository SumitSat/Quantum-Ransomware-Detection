import os
import argparse
import subprocess

def run_step(script, args=""):
    print(f"\n{'='*60}")
    print(f"[RUNNING] {script} {args}")
    print(f"{'='*60}")
    cmd = f"python {script} {args}".strip()
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Q-TERD Unified Execution Pipeline")
    parser.add_argument("--device", type=str, default="lightning.kokkos", help="PennyLane backend (default.qubit, lightning.gpu, or lightning.kokkos)")
    parser.add_argument("--epochs", type=int, default=10, help="Training Epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch Size")
    args = parser.parse_args()

    # Step 1: Synthesis
    if not os.path.exists("data/synthetic/synthetic_ransomware_dataset.csv"):
        print("[INIT] Synthetic dataset not found. Generating...")
        run_step("src/data_loader.py")
    else:
        print("[SKIP] Synthetic data exists.")

    # Step 2: A/B Training
    print("\n[PHASE] Training Old Baseline (CNN-VQC)")
    run_step("src/train.py", f"--device {args.device} --epochs {args.epochs} --batch-size {args.batch_size} --model cnn-vqc")

    print("\n[PHASE] Training New Architecture (QLSTM)")
    run_step("src/train.py", f"--device {args.device} --epochs {args.epochs} --batch-size {args.batch_size} --model qlstm")

    # Step 3: SHAP Explainability Graphing 
    print("\n[PHASE] Validating QLSTM Temporal Lead Time via Q-SHAP")
    run_step("src/explain.py", f"--device {args.device} --model qlstm")

    print("\n[SUCCESS] Unified Q-TERD Pipeline Completed! Checkpoints and plots are available.")

if __name__ == "__main__":
    main()
