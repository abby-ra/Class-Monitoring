"""
Configuration file for the Class Monitoring System
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved_models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Dataset configuration
DATASET_PATH = r'D:\Project\Class_Monitoring\archive (7)'  # Update this to your dataset path
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed')
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'train')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'test')
VAL_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'validation')

# Model configuration
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Engagement labels mapping
ENGAGEMENT_LABELS = {
    'engaged': ['engaged', 'interested', 'focused', 'attentive', 'alert'],
    'disengaged': ['disengaged', 'bored', 'drowsy', 'confused', 'frustrated', 'distracted']
}

# Model paths
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'engagement_model.h5')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

# Real-time monitoring
WEBCAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.7
ALERT_THRESHOLD = 0.3  # If engagement score below this, trigger alert

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)