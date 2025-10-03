"""
Data preprocessing utilities for the Class Monitoring System
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from config import *

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
    def map_engagement_labels(self, label):
        """Map original labels to engaged/disengaged"""
        label_lower = label.lower()
        if any(eng_label in label_lower for eng_label in ENGAGEMENT_LABELS['engaged']):
            return 'engaged'
        else:
            return 'disengaged'
    
    def load_and_preprocess_image(self, image_path, target_size=IMAGE_SIZE):
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def create_dataset_from_directory(self, dataset_path):
        """Create dataset from directory structure"""
        image_paths = []
        labels = []
        
        print("Loading dataset...")
        
        # Walk through dataset directory
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    # Get label from parent directory name
                    label = os.path.basename(root)
                    
                    image_paths.append(image_path)
                    labels.append(label)
        
        print(f"Found {len(image_paths)} images")
        
        # Map labels to engagement categories
        engagement_labels = [self.map_engagement_labels(label) for label in labels]
        
        # Create DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'original_label': labels,
            'engagement': engagement_labels
        })
        
        return df
    
    def preprocess_dataset(self, df, save_processed=True):
        """Preprocess entire dataset"""
        print("Preprocessing images...")
        
        processed_images = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            image = self.load_and_preprocess_image(row['image_path'])
            if image is not None:
                processed_images.append(image)
                valid_indices.append(idx)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} images")
        
        # Filter DataFrame to only include valid images
        df_filtered = df.iloc[valid_indices].reset_index(drop=True)
        processed_images = np.array(processed_images)
        
        print(f"Successfully processed {len(processed_images)} images")
        
        return processed_images, df_filtered
    
    def split_dataset(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split dataset into train, validation, and test sets"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def encode_labels(self, labels):
        """Encode labels for model training"""
        self.label_encoder.fit(['engaged', 'disengaged'])
        encoded_labels = self.label_encoder.transform(labels)
        return encoded_labels
    
    def save_label_encoder(self, path=LABEL_ENCODER_PATH):
        """Save label encoder"""
        with open(path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to {path}")
    
    def load_label_encoder(self, path=LABEL_ENCODER_PATH):
        """Load label encoder"""
        with open(path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"Label encoder loaded from {path}")
    
    def get_class_distribution(self, labels):
        """Get class distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print("Class distribution:")
        for label, count in distribution.items():
            print(f"  {label}: {count} samples")
        return distribution

def main():
    """Test data preprocessing"""
    preprocessor = DataPreprocessor()
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path {DATASET_PATH} does not exist!")
        print("Please update DATASET_PATH in config.py")
        return
    
    # Load dataset
    df = preprocessor.create_dataset_from_directory(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")
    print(df['engagement'].value_counts())
    
    # Preprocess images
    X, df_filtered = preprocessor.preprocess_dataset(df)
    y = df_filtered['engagement'].values
    
    # Encode labels
    y_encoded = preprocessor.encode_labels(y)
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(X, y_encoded)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save label encoder
    preprocessor.save_label_encoder()
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()