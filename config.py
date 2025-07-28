"""
configuration varriables.

"""

import torch
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE=8
LEARNING_RATE=0.1
SEED=42
EPOCHS=40
NUM_CLASSES=21
DATASET_DIR="https://www.kaggle.com/datasets/alirezachahardoli/iranian-car-imageclassification"

