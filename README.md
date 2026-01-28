# Student Feedback Analysis using QNLP

## Overview
A Quantum Machine Learning project to analyze student feedback using **Quantum LSTM** (Long Short-Term Memory) networks.

## Features
- **Quantum NLP**: Uses PennyLane for quantum circuit simulation integrated with PyTorch.
- **Hybrid Architecture**: Combines classical word embeddings (8-dim) with a 4-qubit Quantum Recurrent Neural Network.
- **Sentiment Analysis**: Classifies feedback as Positive or Negative to highlight areas for improvement.
- **Synthetic Data**: Custom dataset generator mirroring real-world university feedback forms.

## Project Structure
| File | Description |
|------|-------------|
| `generate_dataset.py` | Creates `student_feedback.csv` with 200 samples. |
| `data_processor.py` | Cleans data and assigns sentiment labels using VADER. |
| `quantum_model.py` | Implementation of the QLSTM cell using PennyLane circuits. |
| `main_qnlp.py` | Training script (Train/Test split, Training Loop, Evaluation). |
| `qnlp_model.pth` | Reference trained model weights. |

## Quick Start

### Prerequisites
- Python 3.8+
- `pip install pennylane torch nltk pandas scikit-learn`

### Running the Analysis
1. **Generate Data** (Optional, file already included):
   ```bash
   python generate_dataset.py
   ```
2. **Train the Model**:
   ```bash
   python main_qnlp.py
   ```
   This script will:
   - Load `student_feedback.csv`.
   - Process text and generate labels.
   - Train the Hybrid QLSTM model.
   - Output accuracy and save the model.

## Model Details
- **Qubits**: 4
- **Encoding**: Angle Encoding (Rx, Ry)
- **Circuit**: BasicEntanglerLayers (Variational)
- **Optimizer**: Adam (LR=0.01)