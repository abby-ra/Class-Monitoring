#!/usr/bin/env python3
"""
🎓 INTEGRATED STUDENT ENGAGEMENT MONITORING SYSTEM
==================================================
Complete system integrating ML models, utilities, and real-time monitoring
Auto-detects available components and uses the best options
"""

import cv2
import numpy as np
import time
import json
import os
import sys
import pickle
from datetime import datetime, timedelta
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add project directories to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_time'))

# Import configuration and utilities
try:
    from config import *
    CONFIG_AVAILABLE = True
    print("✅ Configuration loaded from config.py")
except ImportError:
    CONFIG_AVAILABLE = False
    # Fallback configuration
    IMAGE_SIZE = (256, 256)
    CONFIDENCE_THRESHOLD = 0.7
    ALERT_THRESHOLD = 0.3
    print("⚠️ Using fallback configuration")

# Try to import ML components and check for trained models
def check_ml_availability():
    """Check if ML models are available"""
    try:
        # Check for scikit-learn model first
        model_path = os.path.join('models', 'saved_models', 'engagement_model.pkl')
        encoder_path = os.path.join('models', 'saved_models', 'label_encoder.pkl')
        
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            # Test loading the model
            import pickle
            with open(model_path, 'rb') as f:
                test_model = pickle.load(f)
            print("✅ Scikit-learn ML model found and loadable")
            return True
            
        # Check for TensorFlow model as fallback
        tf_model_path = os.path.join('models', 'saved_models', 'engagement_model.h5')
        if os.path.exists(tf_model_path) and os.path.exists(encoder_path):
            try:
                import tensorflow as tf
                test_model = tf.keras.models.load_model(tf_model_path)
                print("✅ TensorFlow ML model found and loadable")
                return True
            except ImportError:
                print("⚠️ TensorFlow model found but TensorFlow not installed")
                
        print("⚠️ No trained ML models found. Run 'python train_quick_model.py' to create them.")
        return False
        
    except Exception as e:
        print(f"⚠️ ML check error: {e}")
        return False

ML_AVAILABLE = check_ml_availability()

# Try to import utilities
try:
    from utils.data_preprocessing import DataPreprocessor
    from utils.evaluation import ModelEvaluator
    UTILS_AVAILABLE = True
    print("✅ Utility modules loaded")
except ImportError:
    UTILS_AVAILABLE = False
    print("⚠️ Utility modules not available")

# Try to import real-time components
try:
    from real_time.webcam_monitor import RealTimeMonitor
    from real_time.alert_system import AlertSystem
    REALTIME_AVAILABLE = True
    print("✅ Real-time monitoring modules loaded")
except ImportError:
    REALTIME_AVAILABLE = False
    print("⚠️ Real-time modules not available")

class IntegratedEngagementDetector:
    """Integrated engagement detector with ML and OpenCV options"""
    
    def __init__(self):
        # Initialize detection modes
        self.ml_model = None
        self.ml_label_encoder = None
        self.detection_mode = "opencv"  # Default to OpenCV
        
        # Load OpenCV classifiers (always available as fallback)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Enhanced parameters from config or defaults
        if CONFIG_AVAILABLE:
            self.engagement_threshold = CONFIDENCE_THRESHOLD
            self.confidence_threshold = ALERT_THRESHOLD
            self.image_size = IMAGE_SIZE
        else:
            self.engagement_threshold = 0.6
            self.confidence_threshold = 0.4
            self.image_size = (256, 256)
        
        # Try to load ML model if available
        self._try_load_ml_model()
        
        # Initialize data preprocessor if available
        if UTILS_AVAILABLE:
            try:
                self.preprocessor = DataPreprocessor()
                print("✅ Data preprocessor initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize preprocessor: {e}")
                self.preprocessor = None
        else:
            self.preprocessor = None
    
    def _try_load_ml_model(self):
        """Try to load trained ML model"""
        if not ML_AVAILABLE:
            return
        
        try:
            # Try to load scikit-learn model first
            model_path = os.path.join('models', 'saved_models', 'engagement_model.pkl')
            encoder_path = os.path.join('models', 'saved_models', 'label_encoder.pkl')
            
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                import pickle
                with open(model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                with open(encoder_path, 'rb') as f:
                    self.ml_label_encoder = pickle.load(f)
                print(f"✅ Scikit-learn ML model loaded from {model_path}")
                self.detection_mode = "ml"
                return
                
            # Fallback to TensorFlow model
            if CONFIG_AVAILABLE and os.path.exists(MODEL_SAVE_PATH):
                try:
                    import tensorflow as tf
                    self.ml_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
                    print(f"✅ TensorFlow ML model loaded from {MODEL_SAVE_PATH}")
                    self.detection_mode = "ml"
                    
                    # Load label encoder
                    if os.path.exists(LABEL_ENCODER_PATH):
                        with open(LABEL_ENCODER_PATH, 'rb') as f:
                            self.ml_label_encoder = pickle.load(f)
                        print("✅ Label encoder loaded")
                except ImportError:
                    print("⚠️ TensorFlow not available for model loading")
                
        except Exception as e:
            print(f"⚠️ Could not load ML model: {e}")
            print("Using OpenCV detection as fallback")
    
    def get_detection_info(self):
        """Get current detection mode information"""
        return {
            'mode': self.detection_mode,
            'ml_available': self.ml_model is not None,
            'config_loaded': CONFIG_AVAILABLE,
            'utils_available': UTILS_AVAILABLE
        }
        
    def detect_engagement(self, frame):
        """Detect student engagement using best available method"""
        if self.detection_mode == "ml" and self.ml_model is not None:
            return self._detect_engagement_ml(frame)
        else:
            return self._detect_engagement_opencv(frame)
    
    def _detect_engagement_ml(self, frame):
        """ML-based engagement detection"""
        try:
            # Extract features from frame (for scikit-learn model)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Calculate features
            if len(faces) > 0:
                # Get largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                
                # Feature extraction
                face_size = (w * h) / (frame.shape[0] * frame.shape[1])  # Relative face size
                
                # Eye detection
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                eye_count = len(eyes)
                
                # Position score
                face_center_x = (x + w/2) / frame.shape[1]
                position_score = 1.0 - abs(face_center_x - 0.5) * 2
                
                # Brightness and contrast
                roi_color = frame[y:y+h, x:x+w]
                brightness = np.mean(roi_color) / 255.0
                contrast = np.std(roi_color) / 255.0
                
                features = np.array([[face_size, eye_count, position_score, brightness, contrast]])
            else:
                # No face detected - default features
                features = np.array([[0.0, 0, 0.0, 0.0, 0.0]])
            
            # Make prediction
            if hasattr(self.ml_model, 'predict_proba'):
                # Scikit-learn model
                prediction_proba = self.ml_model.predict_proba(features)[0]
                predicted_class = np.argmax(prediction_proba)
                confidence = np.max(prediction_proba)
            else:
                # TensorFlow model (fallback)
                processed_frame = cv2.resize(frame, self.image_size)
                processed_frame = processed_frame.astype(np.float32) / 255.0
                processed_frame = np.expand_dims(processed_frame, axis=0)
                prediction = self.ml_model.predict(processed_frame, verbose=0)
                
                if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction)
                else:
                    predicted_class = int(prediction[0] > 0.5)
                    confidence = float(prediction[0]) if predicted_class == 1 else float(1 - prediction[0])
            
            # Convert to engagement label
            if self.ml_label_encoder is not None:
                engagement = self.ml_label_encoder.inverse_transform([predicted_class])[0]
            else:
                engagement = 'engaged' if predicted_class == 1 else 'disengaged'
            
            # Also get OpenCV face detection for additional details
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw face rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            return {
                'engagement': engagement,
                'confidence': confidence,
                'faces': faces,
                'details': {
                    'face_count': len(faces),
                    'detection_method': 'ml',
                    'ml_prediction': float(prediction[0]),
                    'predicted_class': predicted_class
                }
            }
            
        except Exception as e:
            print(f"ML detection failed: {e}, falling back to OpenCV")
            return self._detect_engagement_opencv(frame)
    
    def _detect_engagement_opencv(self, frame):
        """OpenCV-based engagement detection (original method)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        engagement_score = 0.0
        total_eyes = 0
        face_positions = []
        
        # Draw rectangles and analyze each face
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes in face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            eye_count = len(eyes)
            total_eyes += eye_count
            
            # Draw eye rectangles
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Calculate face position (center of frame = 0.5)
            face_center_x = (x + w/2) / frame.shape[1]
            face_positions.append(face_center_x)
            
            # Eye engagement factor (2 eyes = optimal)
            eye_factor = min(eye_count / 2.0, 1.0)
            
            # Position factor (closer to center = better)
            position_factor = 1.0 - abs(face_center_x - 0.5) * 2
            
            # Size factor (bigger face = closer = more engaged)
            size_factor = min((w * h) / (frame.shape[0] * frame.shape[1] * 0.1), 1.0)
            
            # Combined engagement score
            face_engagement = (eye_factor * 0.5) + (position_factor * 0.3) + (size_factor * 0.2)
            engagement_score = max(engagement_score, face_engagement)
        
        # Determine engagement status
        if engagement_score >= self.engagement_threshold:
            engagement = 'engaged'
            confidence = min(engagement_score, 0.95)
        elif engagement_score >= self.confidence_threshold:
            engagement = 'partially_engaged'
            confidence = engagement_score
        else:
            engagement = 'disengaged'
            confidence = max(engagement_score, 0.1)
        
        # Calculate average position
        avg_position = sum(face_positions) / len(face_positions) if face_positions else 0.0
        
        return {
            'engagement': engagement,
            'confidence': confidence,
            'faces': faces,
            'details': {
                'face_count': len(faces),
                'eye_count': total_eyes,
                'position_score': avg_position,
                'engagement_score': engagement_score,
                'detection_method': 'opencv'
            }
        }

class IntegratedAlertSystem:
    """Integrated alert system with real-time module integration"""
    
    def __init__(self):
        self.alerts = []
        self.engagement_history = []
        self.session_stats = {
            'start_time': datetime.now(),
            'total_readings': 0,
            'engaged_count': 0,
            'partially_engaged_count': 0,
            'disengaged_count': 0
        }
        
        # Load thresholds from config or use defaults
        if CONFIG_AVAILABLE:
            self.alert_config = {
                'disengaged_duration': 10,
                'low_confidence_threshold': ALERT_THRESHOLD,
                'no_face_duration': 8,
                'alert_cooldown': 5
            }
        else:
            self.alert_config = {
                'disengaged_duration': 10,
                'low_confidence_threshold': 0.3,
                'no_face_duration': 8,
                'alert_cooldown': 5
            }
        
        self.last_alerts = {}
        
        # Try to integrate with external alert system
        if REALTIME_AVAILABLE:
            try:
                self.external_alert_system = AlertSystem()
                print("✅ External alert system integrated")
            except Exception as e:
                print(f"⚠️ Could not integrate external alert system: {e}")
                self.external_alert_system = None
        else:
            self.external_alert_system = None
    
    def log_engagement(self, result):
        """Log engagement and trigger smart alerts"""
        now = datetime.now()
        
        # Update session stats
        self.session_stats['total_readings'] += 1
        engagement = result['engagement']
        
        if engagement == 'engaged':
            self.session_stats['engaged_count'] += 1
        elif engagement == 'partially_engaged':
            self.session_stats['partially_engaged_count'] += 1
        else:
            self.session_stats['disengaged_count'] += 1
        
        # Store history
        engagement_data = {
            'timestamp': now.isoformat(),
            'engagement': engagement,
            'confidence': result['confidence'],
            'details': result['details']
        }
        self.engagement_history.append(engagement_data)
        
        # Keep only recent history
        if len(self.engagement_history) > 500:
            self.engagement_history.pop(0)
        
        # Log to external alert system if available
        if self.external_alert_system:
            try:
                self.external_alert_system.log_engagement(engagement, result['confidence'])
            except Exception as e:
                print(f"External alert system error: {e}")
        
        # Check for alerts
        self._check_smart_alerts(result, now)
        
        # Save engagement data periodically
        if self.session_stats['total_readings'] % 50 == 0:
            self._save_session_data()
    
    def _check_smart_alerts(self, result, timestamp):
        """Smart alert checking with cooldown"""
        engagement = result['engagement']
        confidence = result['confidence']
        face_count = result['details']['face_count']
        
        # Check sustained disengagement
        if engagement == 'disengaged':
            if self._should_alert('disengaged', timestamp):
                recent_disengaged = sum(1 for entry in self.engagement_history[-10:]
                                      if entry['engagement'] == 'disengaged')
                if recent_disengaged >= 8:  # 8 out of last 10 readings
                    self._trigger_alert('DISENGAGED', 
                                      '🔴 Student appears disengaged', 
                                      timestamp, 'high')
        
        # Check no student detected
        if face_count == 0:
            if self._should_alert('no_student', timestamp):
                self._trigger_alert('NO_STUDENT', 
                                  '👤 No student detected', 
                                  timestamp, 'medium')
        
        # Check low confidence
        if confidence < self.alert_config['low_confidence_threshold']:
            if self._should_alert('low_confidence', timestamp):
                self._trigger_alert('LOW_CONFIDENCE', 
                                  f'📊 Detection confidence low: {confidence:.2f}', 
                                  timestamp, 'low')
    
    def _should_alert(self, alert_type, timestamp):
        """Check if enough time has passed since last alert of this type"""
        if alert_type not in self.last_alerts:
            return True
        
        time_diff = (timestamp - self.last_alerts[alert_type]).seconds
        return time_diff >= self.alert_config['alert_cooldown']
    
    def _trigger_alert(self, alert_type, message, timestamp, priority):
        """Trigger a smart alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp.isoformat(),
            'priority': priority
        }
        
        self.alerts.append(alert)
        self.last_alerts[alert_type] = timestamp
        
        # Print with emoji
        emoji = {'high': '🚨', 'medium': '⚠️', 'low': 'ℹ️'}
        print(f"\n{emoji[priority]} {message}")
        
        return alert
    
    def get_session_summary(self):
        """Get comprehensive session summary"""
        if self.session_stats['total_readings'] == 0:
            return {}
        
        total = self.session_stats['total_readings']
        duration = (datetime.now() - self.session_stats['start_time']).total_seconds()
        
        engagement_rate = (self.session_stats['engaged_count'] / total) * 100
        partial_rate = (self.session_stats['partially_engaged_count'] / total) * 100
        
        return {
            'duration_seconds': duration,
            'total_readings': total,
            'engagement_rate': engagement_rate,
            'partial_engagement_rate': partial_rate,
            'total_alerts': len(self.alerts),
            'session_start': self.session_stats['start_time'].isoformat(),
            'integration_info': {
                'config_loaded': CONFIG_AVAILABLE,
                'ml_available': ML_AVAILABLE,
                'utils_available': UTILS_AVAILABLE,
                'realtime_available': REALTIME_AVAILABLE
            }
        }
    
    def _save_session_data(self):
        """Save session data using utilities if available"""
        try:
            if CONFIG_AVAILABLE and LOGS_DIR:
                os.makedirs(LOGS_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(LOGS_DIR, f'session_{timestamp}.json')
                
                session_data = {
                    'session_summary': self.get_session_summary(),
                    'engagement_history': self.engagement_history[-100:],  # Last 100 entries
                    'alerts': self.alerts
                }
                
                with open(log_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                print(f"📄 Session data saved to {log_file}")
        except Exception as e:
            print(f"Could not save session data: {e}")
    
    def get_evaluation_metrics(self):
        """Get evaluation metrics using utility classes if available"""
        if not UTILS_AVAILABLE or len(self.engagement_history) < 10:
            return {}
        
        try:
            evaluator = ModelEvaluator()
            
            # Prepare data for evaluation
            engagements = [entry['engagement'] for entry in self.engagement_history]
            confidences = [entry['confidence'] for entry in self.engagement_history]
            
            # Basic statistics
            total_readings = len(engagements)
            engaged_count = engagements.count('engaged')
            engagement_rate = (engaged_count / total_readings) * 100
            avg_confidence = sum(confidences) / len(confidences)
            
            return {
                'total_readings': total_readings,
                'engagement_rate': engagement_rate,
                'average_confidence': avg_confidence,
                'engagement_distribution': {
                    'engaged': engagements.count('engaged'),
                    'partially_engaged': engagements.count('partially_engaged'),
                    'disengaged': engagements.count('disengaged')
                }
            }
        except Exception as e:
            print(f"Could not generate evaluation metrics: {e}")
            return {}

class IntegratedMonitoringApp:
    """Integrated monitoring app with all components connected"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎓 Integrated Student Engagement Monitor")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize integrated components
        self.detector = IntegratedEngagementDetector()
        self.alert_system = IntegratedAlertSystem()
        self.cap = None
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Get detection info for display
        self.detection_info = self.detector.get_detection_info()
        
        # Setup UI
        self.setup_ui()
        
        print("🎓 Integrated Monitoring System Initialized")
        print(f"📊 Detection Mode: {self.detection_info['mode'].upper()}")
        print(f"🧠 ML Available: {self.detection_info['ml_available']}")
        print(f"⚙️ Config Loaded: {self.detection_info['config_loaded']}")
        print(f"🛠️ Utils Available: {self.detection_info['utils_available']}")
        
    def setup_ui(self):
        """Setup the streamlined UI"""
        # Title with integration status
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=100)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="🎓 INTEGRATED STUDENT ENGAGEMENT MONITOR", 
                              font=("Arial", 20, "bold"),
                              fg='white', bg='#2c3e50')
        title_label.pack(pady=(5, 0))
        
        # Status indicators
        status_text = f"🧠 {self.detection_info['mode'].upper()} | "
        status_text += f"ML: {'✅' if self.detection_info['ml_available'] else '❌'} | "
        status_text += f"Config: {'✅' if CONFIG_AVAILABLE else '❌'} | "
        status_text += f"Utils: {'✅' if UTILS_AVAILABLE else '❌'}"
        
        status_label = tk.Label(title_frame, 
                               text=status_text,
                               font=("Arial", 10),
                               fg='#ecf0f1', bg='#2c3e50')
        status_label.pack()
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Controls
        self.setup_controls(main_frame)
        
        # Status display
        self.setup_status(main_frame)
        
        # Chart
        self.setup_chart(main_frame)
    
    def setup_controls(self, parent):
        """Setup control buttons"""
        control_frame = tk.Frame(parent, bg='#f0f0f0')
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Big start button
        self.start_btn = tk.Button(control_frame, 
                                  text="▶ START MONITORING", 
                                  font=("Arial", 16, "bold"),
                                  bg='#27ae60', fg='white',
                                  command=self.start_monitoring,
                                  height=2, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_btn = tk.Button(control_frame, 
                                 text="⏹ STOP", 
                                 font=("Arial", 16, "bold"),
                                 bg='#e74c3c', fg='white',
                                 command=self.stop_monitoring,
                                 height=2, width=15,
                                 state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Detection mode switch button (if ML available)
        if self.detection_info['ml_available']:
            self.mode_btn = tk.Button(control_frame,
                                     text="🔄 Switch to ML" if self.detection_info['mode'] == 'opencv' else "🔄 Switch to OpenCV",
                                     font=("Arial", 12, "bold"),
                                     bg='#3498db', fg='white',
                                     command=self.switch_detection_mode,
                                     height=2, width=15)
            self.mode_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Status indicator
        self.status_var = tk.StringVar(value="Ready to start monitoring")
        status_label = tk.Label(control_frame, 
                               textvariable=self.status_var,
                               font=("Arial", 14),
                               bg='#f0f0f0')
        status_label.pack(side=tk.LEFT, padx=20)
    
    def setup_status(self, parent):
        """Setup real-time status display"""
        status_frame = tk.LabelFrame(parent, text="📊 Live Status", 
                                   font=("Arial", 14, "bold"),
                                   bg='#f0f0f0')
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Create status grid
        self.status_labels = {}
        
        # Row 1
        tk.Label(status_frame, text="Engagement:", font=("Arial", 12, "bold"), 
                bg='#f0f0f0').grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.status_labels['engagement'] = tk.Label(status_frame, text="Unknown", 
                                                   font=("Arial", 12), bg='#f0f0f0')
        self.status_labels['engagement'].grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        tk.Label(status_frame, text="Confidence:", font=("Arial", 12, "bold"), 
                bg='#f0f0f0').grid(row=0, column=2, sticky=tk.W, padx=10, pady=5)
        self.status_labels['confidence'] = tk.Label(status_frame, text="0.00", 
                                                   font=("Arial", 12), bg='#f0f0f0')
        self.status_labels['confidence'].grid(row=0, column=3, sticky=tk.W, padx=10, pady=5)
        
        # Row 2
        tk.Label(status_frame, text="Session Time:", font=("Arial", 12, "bold"), 
                bg='#f0f0f0').grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.status_labels['time'] = tk.Label(status_frame, text="00:00:00", 
                                             font=("Arial", 12), bg='#f0f0f0')
        self.status_labels['time'].grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
        
        tk.Label(status_frame, text="Engagement Rate:", font=("Arial", 12, "bold"), 
                bg='#f0f0f0').grid(row=1, column=2, sticky=tk.W, padx=10, pady=5)
        self.status_labels['rate'] = tk.Label(status_frame, text="0%", 
                                             font=("Arial", 12), bg='#f0f0f0')
        self.status_labels['rate'].grid(row=1, column=3, sticky=tk.W, padx=10, pady=5)
        
        # Row 3 - Detection method
        tk.Label(status_frame, text="Detection Method:", font=("Arial", 12, "bold"), 
                bg='#f0f0f0').grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.status_labels['method'] = tk.Label(status_frame, text=self.detection_info['mode'].upper(), 
                                               font=("Arial", 12), bg='#f0f0f0')
        self.status_labels['method'].grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)
        
        if UTILS_AVAILABLE:
            tk.Label(status_frame, text="Integration:", font=("Arial", 12, "bold"), 
                    bg='#f0f0f0').grid(row=2, column=2, sticky=tk.W, padx=10, pady=5)
            integration_status = f"ML:{'✅' if ML_AVAILABLE else '❌'} Utils:{'✅' if UTILS_AVAILABLE else '❌'}"
            self.status_labels['integration'] = tk.Label(status_frame, text=integration_status, 
                                                        font=("Arial", 10), bg='#f0f0f0')
            self.status_labels['integration'].grid(row=2, column=3, sticky=tk.W, padx=10, pady=5)
    
    def setup_chart(self, parent):
        """Setup engagement chart"""
        chart_frame = tk.LabelFrame(parent, text="📈 Engagement Over Time", 
                                   font=("Arial", 14, "bold"),
                                   bg='#f0f0f0')
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize empty chart
        self.ax.set_title('Student Engagement Timeline', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Engagement Level')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        
        # Export controls
        export_frame = tk.Frame(parent, bg='#f0f0f0')
        export_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Export button
        export_btn = tk.Button(export_frame,
                              text="📊 Export Session Data",
                              font=("Arial", 12, "bold"),
                              bg='#9b59b6', fg='white',
                              command=self.export_session_data,
                              height=1, width=20)
        export_btn.pack(side=tk.LEFT, padx=5)
        
        # Integration info button
        if UTILS_AVAILABLE or ML_AVAILABLE:
            info_btn = tk.Button(export_frame,
                                text="ℹ️ System Info",
                                font=("Arial", 12, "bold"),
                                bg='#34495e', fg='white',
                                command=self.show_system_info,
                                height=1, width=15)
            info_btn.pack(side=tk.LEFT, padx=5)
    
    def start_monitoring(self):
        """Start monitoring"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera!")
                return
            
            self.is_monitoring = True
            self.session_start = time.time()
            
            # Update UI
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self.status_var.set("🟢 Monitoring Active...")
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            # Start UI updates
            self.update_ui()
            
            print("🚀 Student Engagement Monitoring Started!")
            print("📹 Look at the camera for best results")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start: {str(e)}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
        if self.cap:
            self.cap.release()
        
        # Update UI
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("🔴 Monitoring Stopped")
        
        # Show comprehensive summary
        summary = self.alert_system.get_session_summary()
        if summary:
            duration = int(summary['duration_seconds'])
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Enhanced summary with integration info
            summary_text = f"""
🎉 Session Complete!

📊 Session Statistics:
Duration: {hours:02d}:{minutes:02d}:{seconds:02d}
Total Readings: {summary['total_readings']}
Engagement Rate: {summary['engagement_rate']:.1f}%
Total Alerts: {summary['total_alerts']}

🔧 System Integration:
Detection Mode: {self.detection_info['mode'].upper()}
ML Models: {'✅ Active' if self.detection_info['ml_available'] else '❌ Not Available'}
Configuration: {'✅ Loaded' if CONFIG_AVAILABLE else '❌ Using Defaults'}
Utilities: {'✅ Available' if UTILS_AVAILABLE else '❌ Not Available'}
            """
            messagebox.showinfo("Session Summary", summary_text)
            
            # Show evaluation metrics if available
            if UTILS_AVAILABLE:
                metrics = self.alert_system.get_evaluation_metrics()
                if metrics:
                    print("📈 Session Evaluation Metrics:")
                    for key, value in metrics.items():
                        print(f"   {key}: {value}")
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        cv2.namedWindow('Student Engagement Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Student Engagement Monitor', 800, 600)
        
        while self.is_monitoring:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Detect engagement
                result = self.detector.detect_engagement(frame)
                
                # Log to alert system
                self.alert_system.log_engagement(result)
                
                # Store for UI
                self.current_result = result
                
                # Add status text to frame
                engagement = result['engagement']
                confidence = result['confidence']
                
                # Color based on engagement
                if engagement == 'engaged':
                    color = (0, 255, 0)  # Green
                    status_text = f"ENGAGED ({confidence:.2f})"
                elif engagement == 'partially_engaged':
                    color = (0, 255, 255)  # Yellow
                    status_text = f"PARTIALLY ENGAGED ({confidence:.2f})"
                else:
                    color = (0, 0, 255)  # Red
                    status_text = f"DISENGAGED ({confidence:.2f})"
                
                # Add text overlay
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Show frame
                cv2.imshow('Student Engagement Monitor', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
        
        cv2.destroyAllWindows()
    
    def update_ui(self):
        """Update UI elements"""
        if self.is_monitoring and hasattr(self, 'current_result'):
            result = self.current_result
            
            # Update status labels
            engagement = result['engagement']
            confidence = result['confidence']
            
            # Engagement with color
            engagement_text = engagement.replace('_', ' ').title()
            color = '#27ae60' if engagement == 'engaged' else '#e74c3c' if engagement == 'disengaged' else '#f39c12'
            self.status_labels['engagement'].configure(text=engagement_text, fg=color)
            self.status_labels['confidence'].configure(text=f"{confidence:.2f}")
            
            # Session time
            duration = int(time.time() - self.session_start)
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.status_labels['time'].configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Engagement rate
            summary = self.alert_system.get_session_summary()
            if summary:
                rate = summary['engagement_rate']
                self.status_labels['rate'].configure(text=f"{rate:.1f}%")
            
            # Update chart every 10 readings
            if len(self.alert_system.engagement_history) % 10 == 0:
                self.update_chart()
        
        if self.is_monitoring:
            self.root.after(1000, self.update_ui)
    
    def update_chart(self):
        """Update the engagement chart"""
        if not self.alert_system.engagement_history:
            return
        
        # Get recent data
        recent_data = self.alert_system.engagement_history[-50:]  # Last 50 readings
        
        timestamps = []
        values = []
        
        for entry in recent_data:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            timestamps.append(timestamp)
            
            # Convert engagement to numeric
            if entry['engagement'] == 'engaged':
                values.append(1.0)
            elif entry['engagement'] == 'partially_engaged':
                values.append(0.5)
            else:
                values.append(0.0)
        
        # Clear and redraw
        self.ax.clear()
        
        # Plot line
        self.ax.plot(timestamps, values, 'b-', linewidth=2, marker='o', markersize=4)
        self.ax.fill_between(timestamps, values, alpha=0.3)
        
        # Styling
        self.ax.set_title('Student Engagement Timeline', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Engagement Level')
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid(True, alpha=0.3)
        
        # Y-axis labels
        self.ax.set_yticks([0, 0.5, 1])
        self.ax.set_yticklabels(['Disengaged', 'Partial', 'Engaged'])
        
        # Format dates
        self.fig.autofmt_xdate()
        
        self.canvas.draw()
    
    def switch_detection_mode(self):
        """Switch between ML and OpenCV detection modes"""
        if not self.is_monitoring and self.detection_info['ml_available']:
            if self.detector.detection_mode == 'opencv':
                self.detector.detection_mode = 'ml'
                self.mode_btn.configure(text="🔄 Switch to OpenCV")
                self.status_labels['method'].configure(text="ML")
                print("🧠 Switched to ML detection mode")
            else:
                self.detector.detection_mode = 'opencv'
                self.mode_btn.configure(text="🔄 Switch to ML")
                self.status_labels['method'].configure(text="OPENCV")
                print("👁️ Switched to OpenCV detection mode")
            
            # Update detection info
            self.detection_info['mode'] = self.detector.detection_mode
        else:
            if self.is_monitoring:
                messagebox.showwarning("Cannot Switch", "Stop monitoring first to switch detection modes")
            else:
                messagebox.showinfo("ML Not Available", "Machine learning models are not available")
    
    def export_session_data(self):
        """Export comprehensive session data"""
        if not self.alert_system.engagement_history:
            messagebox.showwarning("No Data", "No session data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Session Data"
        )
        
        if filename:
            try:
                # Comprehensive export data
                export_data = {
                    'session_summary': self.alert_system.get_session_summary(),
                    'engagement_history': self.alert_system.engagement_history,
                    'alerts': self.alert_system.alerts,
                    'detection_info': self.detection_info,
                    'system_integration': {
                        'ml_available': ML_AVAILABLE,
                        'config_loaded': CONFIG_AVAILABLE,
                        'utils_available': UTILS_AVAILABLE,
                        'realtime_available': REALTIME_AVAILABLE
                    }
                }
                
                # Add evaluation metrics if available
                if UTILS_AVAILABLE:
                    metrics = self.alert_system.get_evaluation_metrics()
                    if metrics:
                        export_data['evaluation_metrics'] = metrics
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                messagebox.showinfo("Export Complete", f"Session data exported to:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
    
    def show_system_info(self):
        """Show detailed system integration information"""
        info_text = f"""
🎓 INTEGRATED MONITORING SYSTEM INFO

🔧 Core Components:
• Detection Engine: {self.detection_info['mode'].upper()}
• Alert System: Integrated Smart Alerts
• User Interface: Professional Tkinter GUI
• Visualization: Matplotlib Charts

🧠 Machine Learning Integration:
• Status: {'✅ Available' if ML_AVAILABLE else '❌ Not Available'}
• Models: {'CNN-based engagement detection' if ML_AVAILABLE else 'Not loaded'}
• Fallback: OpenCV computer vision

⚙️ Configuration System:
• Status: {'✅ Loaded from config.py' if CONFIG_AVAILABLE else '❌ Using defaults'}
• Settings: {'Centralized configuration' if CONFIG_AVAILABLE else 'Built-in defaults'}

🛠️ Utility Integration:
• Data Processing: {'✅ Available' if UTILS_AVAILABLE else '❌ Not available'}
• Model Evaluation: {'✅ Available' if UTILS_AVAILABLE else '❌ Not available'}
• Data Augmentation: {'✅ Available' if UTILS_AVAILABLE else '❌ Not available'}

🎥 Real-time Monitoring:
• External Modules: {'✅ Integrated' if REALTIME_AVAILABLE else '❌ Not available'}
• Alert System: {'Enhanced integration' if REALTIME_AVAILABLE else 'Built-in system'}

📊 Current Session:
• Total Readings: {len(self.alert_system.engagement_history)}
• Detection Method: {self.detection_info['mode'].upper()}
• Alerts Generated: {len(self.alert_system.alerts)}
        """
        
        # Create info window
        info_window = tk.Toplevel(self.root)
        info_window.title("System Integration Information")
        info_window.geometry("600x500")
        info_window.configure(bg='#f0f0f0')
        
        # Text widget with scrollbar
        text_frame = tk.Frame(info_window, bg='#f0f0f0')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Courier", 10),
                             bg='white', fg='black')
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(tk.END, info_text)
        text_widget.configure(state=tk.DISABLED)
        
        # Close button
        close_btn = tk.Button(info_window, text="Close", command=info_window.destroy,
                             font=("Arial", 12), bg='#e74c3c', fg='white')
        close_btn.pack(pady=10)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("=" * 70)
    print("🎓 INTEGRATED STUDENT ENGAGEMENT MONITORING SYSTEM")
    print("=" * 70)
    print("✅ Multi-component integration system")
    print("✅ ML models + OpenCV + Real-time monitoring")
    print("✅ Smart configuration and utilities")
    print("✅ Professional GUI with advanced features")
    print("=" * 70)
    
    # Show integration status
    print("🔍 System Integration Status:")
    print(f"   🧠 Machine Learning: {'✅ Available' if ML_AVAILABLE else '❌ Not Available'}")
    print(f"   ⚙️ Configuration: {'✅ Loaded' if CONFIG_AVAILABLE else '❌ Using Defaults'}")
    print(f"   🛠️ Utilities: {'✅ Available' if UTILS_AVAILABLE else '❌ Not Available'}")
    print(f"   🎥 Real-time Modules: {'✅ Available' if REALTIME_AVAILABLE else '❌ Not Available'}")
    print("=" * 70)
    print("🚀 Launching integrated monitoring system...")
    print("=" * 70)
    
    app = IntegratedMonitoringApp()
    app.run()

if __name__ == "__main__":
    main()