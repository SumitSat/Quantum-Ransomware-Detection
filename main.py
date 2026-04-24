import os
from src.train import train
from src.evaluate import evaluate
import argparse

def main():
    parser = argparse.ArgumentParser(description="Q-TERD VERA Quantum Static Detection")
    parser.add_argument("--device", type=str, default="lightning.kokkos", help="PennyLane backend")
    parser.add_argument("--epochs", type=int, default=1, help="Training Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch Size")
    args = parser.parse_args()

    # 1. Train the model across 5-Folds
    print(">>> 1. Training VERA Model (5-Fold CV) <<<")
    train(device_string=args.device, epochs=args.epochs, batch_size=args.batch_size)

    # 2. Evaluate Fold 1 as basic check (Detailed evaluation is done during train script natively)
    print("\n>>> 2. Final Evaluation Verification (Fold 1) <<<")
    evaluate(device_string=args.device, fold=1)
    
    print("\n✅ Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
