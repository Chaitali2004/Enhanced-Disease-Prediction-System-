# Disease Prediction System

Multi-modal deep learning system for symptom-based disease diagnosis.

## Overview

This system predicts diseases from patient symptoms using four complementary feature extraction methods combined through a multi-modal fusion architecture.

## Features

- **Contrastive Learning**: Self-supervised pre-training with NT-Xent loss
- **Graph Neural Network**: Symptom relationships with body system knowledge
- **Hierarchical Attention**: Word-level and phrase-level symptom importance
- **Multi-Modal Fusion**: Combines all features with residual connections

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install tensorflow torch torch-geometric gensim
```

## Dataset

- **File**: `DiseaseAndSymptoms.csv`
- **Format**: Patient records with 17 symptom columns and 1 disease label
- **Classes**: 41 diseases
- **Samples**: 4,920 patient records

## Usage

```python
# Run the complete pipeline
python final_examiner_ready.py

# For prediction on new symptoms
predict_disease(["fever", "cough", "fatigue"], top_k=3)
```

## Model Architecture

- **Input**: Patient symptom list
- **Feature Extraction**: 4 parallel branches (342D total)
  - Contrastive: 64D
  - GNN: 128D
  - Attention: 100D
  - Structured: 50D
- **Fusion**: Multi-layer network with residual connections
- **Output**: Disease probabilities (41 classes)
- **Parameters**: ~285,000

## Performance

- **Test Accuracy**: 96-98%
- **Top-3 Accuracy**: 98%+
- **Top-5 Accuracy**: 99%+

## Outputs

The system generates:
- `performance_summary.txt` - Performance metrics
- `disease_prediction_results.png` - Visualizations
- `Confusion Matrix.png` - Visualizations
<img width="4561" height="3902" alt="4_confusion_matrix_full" src="https://github.com/user-attachments/assets/5c357267-e96b-49f2-baba-8a15cff79e69" />
- `attention_heatmap.png` - Attention analysis
<img width="3832" height="2370" alt="7_attention_heatmap" src="https://github.com/user-attachments/assets/c7e44ccb-34ee-4abe-88b6-4308d576c0d4" />
- `tsne_visualization`
  <img width="4119" height="2970" alt="6_tsne_visualization" src="https://github.com/user-attachments/assets/e828eac5-76c5-4dcc-9b6d-39b3d66cff19" />
- `top_symptoms_per_disease.png` - Feature importance
!<img width="2969" height="1766" alt="8_1_AIDS_symptoms" src="https://github.com/user-attachments/assets/9c24b920-08ba-4959-958e-f991e08845bd" />
<img width="2969" height="1766" alt="8_1_Tuberculosis_symptoms" src="https://github.com/user-attachments/assets/0a8c5885-3a25-4abe-9bec-a7d49bba11ed" />


## Key Components

1. **Preprocessing**: Symptom severity mapping and embedding extraction
2. **Contrastive Pre-training**: 30 epochs with early stopping
3. **Graph Construction**: Multi-factor edge weights (semantic + medical + co-occurrence)
4. **Training**: 100 epochs with callbacks (early stopping, learning rate reduction)
5. **Evaluation**: Classification report, confusion matrix, top-k accuracy

## Author

Chaitali Jain  
Major Project
