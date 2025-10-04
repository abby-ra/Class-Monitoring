"""
Quick ML Model Training for Student Engagement Detection
Creates a simple trained model so ML integration works
"""

import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json
from datetime import datetime

def create_synthetic_dataset(num_samples=1000):
    """Create synthetic engagement data for quick training"""
    print("🔄 Creating synthetic training dataset...")
    
    # Create synthetic features (face detection metrics)
    np.random.seed(42)
    
    # Features: [face_size, eye_count, position_score, brightness, contrast]
    engaged_features = []
    disengaged_features = []
    
    # Generate engaged samples (good metrics)
    for _ in range(num_samples // 2):
        face_size = np.random.normal(0.15, 0.05)  # Larger faces (closer to camera)
        eye_count = np.random.choice([2, 2, 2, 1], p=[0.8, 0.1, 0.05, 0.05])  # Mostly 2 eyes
        position_score = np.random.normal(0.5, 0.1)  # Centered position
        brightness = np.random.normal(0.6, 0.1)  # Good lighting
        contrast = np.random.normal(0.7, 0.1)  # Good contrast
        
        engaged_features.append([
            max(0, min(1, face_size)),
            eye_count,
            max(0, min(1, position_score)),
            max(0, min(1, brightness)),
            max(0, min(1, contrast))
        ])
    
    # Generate disengaged samples (poor metrics)
    for _ in range(num_samples // 2):
        face_size = np.random.normal(0.08, 0.03)  # Smaller faces (further away)
        eye_count = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])  # Often no/few eyes
        position_score = np.random.normal(0.3, 0.15)  # Off-center
        brightness = np.random.normal(0.4, 0.15)  # Poor lighting
        contrast = np.random.normal(0.5, 0.15)  # Poor contrast
        
        disengaged_features.append([
            max(0, min(1, face_size)),
            eye_count,
            max(0, min(1, position_score)),
            max(0, min(1, brightness)),
            max(0, min(1, contrast))
        ])
    
    # Combine data
    X = np.array(engaged_features + disengaged_features)
    y = ['engaged'] * len(engaged_features) + ['disengaged'] * len(disengaged_features)
    
    print(f"✅ Created {len(X)} synthetic samples")
    return X, y

def train_simple_model():
    """Train a simple model using scikit-learn (no TensorFlow needed)"""
    print("🧠 Training engagement detection model...")
    
    # Create dataset
    X, y = create_synthetic_dataset()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train a simple model (Random Forest)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained with accuracy: {accuracy:.3f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return model, le

def save_model(model, label_encoder):
    """Save the trained model and label encoder"""
    print("💾 Saving model and label encoder...")
    
    # Create directories
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Save model using pickle (works with scikit-learn)
    model_path = 'models/saved_models/engagement_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save label encoder
    encoder_path = 'models/saved_models/label_encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save model info
    model_info = {
        'model_type': 'RandomForestClassifier',
        'features': ['face_size', 'eye_count', 'position_score', 'brightness', 'contrast'],
        'classes': label_encoder.classes_.tolist(),
        'trained_date': datetime.now().isoformat(),
        'accuracy': 'Synthetic training data'
    }
    
    info_path = 'models/saved_models/model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Label encoder saved to: {encoder_path}")
    print(f"✅ Model info saved to: {info_path}")

def main():
    """Main training function"""
    print("=" * 60)
    print("🧠 QUICK ML MODEL TRAINING")
    print("=" * 60)
    print("Creating a trained model for ML integration...")
    print("=" * 60)
    
    try:
        # Train model
        model, le = train_simple_model()
        
        # Save model
        save_model(model, le)
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! ML model is now available")
        print("=" * 60)
        print("✅ Your student_monitor.py will now show 'ML Available: True'")
        print("✅ You can switch to ML mode in the application")
        print("✅ The model uses facial features for engagement detection")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("💡 Falling back to OpenCV-only mode")

if __name__ == "__main__":
    main()