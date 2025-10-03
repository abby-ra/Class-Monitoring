"""
CNN Model for Student Engagement Classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import numpy as np
from config import *

class EngagementCNN:
    def __init__(self, input_shape=(256, 256, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build CNN model architecture"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=LEARNING_RATE):
        """Compile the model"""
        if self.model is None:
            self.build_model()
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def get_callbacks(self, model_save_path=MODEL_SAVE_PATH):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the model"""
        if self.model is None:
            self.compile_model()
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        print("Starting model training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val_cat),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert labels to categorical
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        # Evaluate
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test_cat, verbose=0
        )
        
        # Calculate F1 score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': f1_score
        }
        
        print("Test Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        return results
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_single(self, image):
        """Predict single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)
        return prediction[0]
    
    def save_model(self, path=MODEL_SAVE_PATH):
        """Save the model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=MODEL_SAVE_PATH):
        """Load a saved model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        
        return self.model.summary()

def main():
    """Test model creation"""
    # Create model
    cnn_model = EngagementCNN()
    
    # Build and compile model
    model = cnn_model.compile_model()
    
    # Print model summary
    print("Model Architecture:")
    cnn_model.get_model_summary()
    
    print(f"Total parameters: {model.count_params():,}")

if __name__ == "__main__":
    main()