"""
Real-time engagement monitoring using webcam
"""

import cv2
import numpy as np
import pickle
from tensorflow import keras
import time
import threading
from collections import deque
from config import *

class RealTimeMonitor:
    def __init__(self, model_path=MODEL_SAVE_PATH, label_encoder_path=LABEL_ENCODER_PATH):
        self.model = None
        self.label_encoder = None
        self.cap = None
        self.is_running = False
        self.engagement_history = deque(maxlen=30)  # Store last 30 predictions
        self.current_engagement = "Unknown"
        self.confidence = 0.0
        
        # Load model and label encoder
        self.load_model(model_path)
        self.load_label_encoder(label_encoder_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the model first using train_model.py")
    
    def load_label_encoder(self, label_encoder_path):
        """Load label encoder"""
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"Label encoder loaded from {label_encoder_path}")
        except Exception as e:
            print(f"Error loading label encoder: {e}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for prediction"""
        # Resize frame
        resized = cv2.resize(frame, IMAGE_SIZE)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch_frame = np.expand_dims(normalized, axis=0)
        
        return batch_frame
    
    def predict_engagement(self, frame):
        """Predict engagement from frame"""
        if self.model is None:
            return "Model not loaded", 0.0
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Make prediction
        prediction = self.model.predict(processed_frame, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx]
        
        # Convert to label
        if self.label_encoder is not None:
            predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        else:
            predicted_label = "engaged" if predicted_class_idx == 1 else "disengaged"
        
        return predicted_label, confidence
    
    def update_engagement_history(self, engagement, confidence):
        """Update engagement history for smoothing"""
        self.engagement_history.append((engagement, confidence))
        
        # Calculate average engagement over recent frames
        if len(self.engagement_history) > 0:
            recent_engagements = [eng for eng, conf in self.engagement_history if conf > CONFIDENCE_THRESHOLD]
            if recent_engagements:
                # Most common engagement in recent history
                engaged_count = recent_engagements.count('engaged')
                disengaged_count = recent_engagements.count('disengaged')
                
                if engaged_count > disengaged_count:
                    self.current_engagement = "engaged"
                else:
                    self.current_engagement = "disengaged"
                
                # Average confidence
                self.confidence = np.mean([conf for eng, conf in self.engagement_history])
    
    def draw_overlay(self, frame):
        """Draw engagement information on frame"""
        height, width = frame.shape[:2]
        
        # Create overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Set color based on engagement
        if self.current_engagement == "engaged":
            color = (0, 255, 0)  # Green
        elif self.current_engagement == "disengaged":
            color = (0, 0, 255)  # Red
        else:
            color = (255, 255, 255)  # White
        
        # Draw text
        cv2.putText(frame, f"Engagement: {self.current_engagement.upper()}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {self.confidence:.2f}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Press 'q' to quit", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Press 's' for screenshot", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw engagement indicator
        indicator_x = width - 100
        indicator_y = 50
        if self.current_engagement == "engaged":
            cv2.circle(frame, (indicator_x, indicator_y), 20, (0, 255, 0), -1)
        elif self.current_engagement == "disengaged":
            cv2.circle(frame, (indicator_x, indicator_y), 20, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (indicator_x, indicator_y), 20, (128, 128, 128), -1)
        
        return frame
    
    def save_screenshot(self, frame):
        """Save screenshot with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"engagement_screenshot_{timestamp}.jpg"
        filepath = os.path.join(DATA_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"Screenshot saved: {filepath}")
    
    def start_monitoring(self, camera_index=WEBCAM_INDEX):
        """Start real-time monitoring"""
        if self.model is None:
            print("Model not loaded. Cannot start monitoring.")
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        print("Starting real-time engagement monitoring...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Mirror the frame horizontally for better user experience
            frame = cv2.flip(frame, 1)
            
            # Predict engagement every few frames to improve performance
            if frame_count % 3 == 0:  # Predict every 3rd frame
                engagement, confidence = self.predict_engagement(frame)
                self.update_engagement_history(engagement, confidence)
            
            # Draw overlay
            frame_with_overlay = self.draw_overlay(frame)
            
            # Display frame
            cv2.imshow('Student Engagement Monitor', frame_with_overlay)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(frame_with_overlay)
            
            frame_count += 1
            
            # Calculate and display FPS every 100 frames
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
        
        self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Monitoring stopped")
    
    def get_engagement_stats(self):
        """Get engagement statistics"""
        if len(self.engagement_history) == 0:
            return {}
        
        engagements = [eng for eng, conf in self.engagement_history]
        engaged_count = engagements.count('engaged')
        disengaged_count = engagements.count('disengaged')
        total_count = len(engagements)
        
        stats = {
            'total_frames': total_count,
            'engaged_frames': engaged_count,
            'disengaged_frames': disengaged_count,
            'engagement_percentage': (engaged_count / total_count) * 100 if total_count > 0 else 0,
            'average_confidence': np.mean([conf for eng, conf in self.engagement_history])
        }
        
        return stats

def main():
    """Test real-time monitoring"""
    monitor = RealTimeMonitor()
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    finally:
        monitor.stop_monitoring()
        
        # Display final stats
        stats = monitor.get_engagement_stats()
        if stats:
            print("\nSession Statistics:")
            print(f"Total frames analyzed: {stats['total_frames']}")
            print(f"Engagement percentage: {stats['engagement_percentage']:.1f}%")
            print(f"Average confidence: {stats['average_confidence']:.2f}")

if __name__ == "__main__":
    main()