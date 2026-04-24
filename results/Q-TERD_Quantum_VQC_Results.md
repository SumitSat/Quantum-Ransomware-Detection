# Q-TERD Model Results: Quantum VQC Pipeline (Path A)
**Date:** April 20, 2026
**Dataset:** VERA (Static PE Features)
**Samples:** 30,652 (15,326 Ransomware | 15,326 Benign)
**Dimensionality:** 550 features scrubbed to 481 (Removed 'label' and 'rich_header_hash')
**Quantum Backend:** default.qubit (8-qubits, Angle Encoding, Variational Setup)
**Cross-Validation:** 5-Fold Stratified (with L2 Regularization & Early Stopping)

## Fold Output
- **Fold 1 Accuracy:** 0.9817
- **Fold 2 Accuracy:** 0.9499
- **Fold 3 Accuracy:** 0.9855
- **Fold 4 Accuracy:** 0.9962
- **Fold 5 Accuracy:** 0.9816
- **Mean Validation Accuracy:** 0.9790

## Detailed Point Metrics (Fold 1 Test Set)
- **Accuracy:**  0.9817
- **Precision:** 0.9774
- **Recall:**    0.9863
- **F1 Score:**  0.9818
- **AUC-ROC:**   0.9927
