"""
Model evaluation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
from config import *

class ModelEvaluator:
    def __init__(self, label_encoder_path=LABEL_ENCODER_PATH):
        self.label_encoder = None
        self.load_label_encoder(label_encoder_path)
    
    def load_label_encoder(self, path):
        """Load label encoder"""
        try:
            with open(path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load label encoder: {e}")
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Convert one-hot encoded labels back to integers if needed
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_int = np.argmax(y_test, axis=1)
        else:
            y_test_int = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_int, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test_int, y_pred, average='weighted')
        
        # Get label names
        if self.label_encoder is not None:
            target_names = self.label_encoder.classes_
        else:
            target_names = ['disengaged', 'engaged']
        
        # Classification report
        report = classification_report(y_test_int, y_pred, target_names=target_names)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_int, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': y_test_int,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'target_names': target_names
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, target_names, title='Confusion Matrix'):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(LOGS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history, title='Training History'):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(LOGS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, target_names):
        """Plot ROC curve for binary classification"""
        if y_pred_proba.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Student Engagement Classification')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(LOGS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            return roc_auc
        else:
            print("ROC curve plotting is only supported for binary classification")
            return None
    
    def print_evaluation_summary(self, results):
        """Print comprehensive evaluation summary"""
        print("=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Weighted Precision: {results['precision']:.4f}")
        print(f"Weighted Recall: {results['recall']:.4f}")
        print(f"Weighted F1-Score: {results['f1_score']:.4f}")
        
        print("\nDetailed Classification Report:")
        print(results['classification_report'])
        
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            results['y_true'], results['y_pred'], average=None
        )
        
        print("\nPer-Class Metrics:")
        for i, class_name in enumerate(results['target_names']):
            print(f"{class_name:>12}: Precision={precision[i]:.4f}, "
                  f"Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    def save_evaluation_report(self, results, filename='evaluation_report.txt'):
        """Save evaluation report to file"""
        filepath = os.path.join(LOGS_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Weighted Precision: {results['precision']:.4f}\n")
            f.write(f"Weighted Recall: {results['recall']:.4f}\n")
            f.write(f"Weighted F1-Score: {results['f1_score']:.4f}\n\n")
            
            f.write("Detailed Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(str(results['confusion_matrix']))
            f.write("\n")
        
        print(f"Evaluation report saved to {filepath}")

def main():
    """Test evaluation functions"""
    evaluator = ModelEvaluator()
    print("Model evaluator initialized successfully!")

if __name__ == "__main__":
    main()