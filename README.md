# LW4_Improving-CNN-Performance
https://colab.research.google.com/drive/19012A2u5flwE4sHLtK6rdKpq_J-q-erz

Metric                          | Baseline (LW3) | Improved (LW4) | Change
--------------------------------|----------------|----------------|--------
ACCURACY & LOSS                 |                |                |
Training Accuracy               | 95.55%         | 99.54%         | +3.99%
Training Loss                   | 0.1485         | 0.0171         | Lower
Validation Accuracy             | 95.54%         | 99.91%         | +4.37%
Validation Loss                 | 0.1333         | 0.0665         | Lower
Generalization Gap (Train−Val)  | 0.01%          | ~0.00%         | ✓ ≤5%
                                |                |                |
CLASSIFICATION METRICS          |                |                |
Precision (macro avg)           | 96.00%         | 99.76%         | +3.76%
Recall (macro avg)              | 96.00%         | 99.71%         | +3.71%
F1-score (macro avg)            | 96.00%         | 99.73%         | +3.73%
                                |                |                |
OVERALL DISCRIMINATION          |                |                |
AUC Score (OvR)                 | 98.49%         | 99.22%         | +0.73%
