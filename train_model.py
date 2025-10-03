"""
Complete training pipeline for Student Engagement Classification
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils.data_preprocessing import DataPreprocessor
from models.cnn_model import EngagementCNN
from utils.evaluation import ModelEvaluator

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("STUDENT ENGAGEMENT CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Initialize components
    print("\n1. Initializing components...")
    preprocessor = DataPreprocessor()
    model = EngagementCNN()
    evaluator = ModelEvaluator()
    
    # Step 2: Check dataset
    print("\n2. Checking dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path {DATASET_PATH} does not exist!")
        print("Please update DATASET_PATH in config.py to point to your dataset")
        print("\nExpected dataset structure:")
        print("dataset/")
        print("├── engaged/")
        print("│   ├── img1.jpg")
        print("│   └── img2.jpg")
        print("└── disengaged/")
        print("    ├── img3.jpg")
        print("    └── img4.jpg")
        return False
    
    # Step 3: Load and preprocess data
    print("\n3. Loading and preprocessing dataset...")
    try:
        # Load dataset
        df = preprocessor.create_dataset_from_directory(DATASET_PATH)
        print(f"Dataset loaded: {len(df)} images found")
        
        # Display class distribution
        print("\nClass distribution:")
        print(df['engagement'].value_counts())
        
        # Check if we have both classes
        unique_classes = df['engagement'].unique()
        if len(unique_classes) < 2:
            print(f"ERROR: Only found {len(unique_classes)} class(es): {unique_classes}")
            print("Need both 'engaged' and 'disengaged' classes for training")
            return False
        
        # Preprocess images
        X, df_filtered = preprocessor.preprocess_dataset(df)
        y = df_filtered['engagement'].values
        
        if len(X) == 0:
            print("ERROR: No valid images found after preprocessing")
            return False
        
        print(f"Successfully preprocessed {len(X)} images")
        
        # Encode labels
        y_encoded = preprocessor.encode_labels(y)
        
        # Split dataset
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(X, y_encoded)
        
        print(f"Dataset split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Save label encoder
        preprocessor.save_label_encoder()
        
    except Exception as e:
        print(f"ERROR in data preprocessing: {e}")
        return False
    
    # Step 4: Build and compile model
    print("\n4. Building and compiling model...")
    try:
        model.compile_model()
        print("Model compiled successfully")
        
        # Print model summary
        print("\nModel Architecture:")
        model.get_model_summary()
        
    except Exception as e:
        print(f"ERROR in model compilation: {e}")
        return False
    
    # Step 5: Train model
    print(f"\n5. Training model for {EPOCHS} epochs...")
    try:
        history = model.train(X_train, y_train, X_val, y_val)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        return False
    
    # Step 6: Evaluate model
    print("\n6. Evaluating model...")
    try:
        # Evaluate on test set
        test_results = model.evaluate(X_test, y_test)
        
        # Comprehensive evaluation
        eval_results = evaluator.evaluate_model(model.model, X_test, y_test)
        
        # Print summary
        evaluator.print_evaluation_summary(eval_results)
        
        # Save evaluation report
        evaluator.save_evaluation_report(eval_results)
        
        # Plot results
        print("\n7. Generating visualizations...")
        
        # Plot training history
        evaluator.plot_training_history(history, "Student Engagement Model - Training History")
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            eval_results['confusion_matrix'], 
            eval_results['target_names'],
            "Student Engagement Classification - Confusion Matrix"
        )
        
        # Plot ROC curve
        roc_auc = evaluator.plot_roc_curve(
            eval_results['y_true'], 
            eval_results['y_pred_proba'], 
            eval_results['target_names']
        )
        
        if roc_auc:
            print(f"ROC AUC Score: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        return False
    
    # Step 7: Save final model
    print("\n8. Saving trained model...")
    try:
        model.save_model()
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"ERROR saving model: {e}")
        return False
    
    # Step 8: Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")
    print(f"Evaluation reports saved to: {LOGS_DIR}")
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"  Precision: {test_results['test_precision']:.4f}")
    print(f"  Recall: {test_results['test_recall']:.4f}")
    print(f"  F1-Score: {test_results['test_f1_score']:.4f}")
    
    print(f"\nTo start real-time monitoring, run:")
    print(f"  python real_time/webcam_monitor.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nTraining failed. Please check the errors above and try again.")
        sys.exit(1)
    else:
        print("\nTraining pipeline completed successfully!")
        sys.exit(0)