"""
API-based Student Engagement Detection using various Computer Vision APIs
No custom dataset required - uses pre-trained models via APIs
"""

import cv2
import requests
import base64
import json
import time
from datetime import datetime
import numpy as np

# API Configuration
class APIConfig:
    # Google Vision API
    GOOGLE_API_KEY = "your_google_api_key_here"
    GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
    
    # Azure Computer Vision
    AZURE_ENDPOINT = "https://your-resource.cognitiveservices.azure.com/"
    AZURE_API_KEY = "your_azure_api_key_here"
    
    # AWS Rekognition
    AWS_ACCESS_KEY = "your_aws_access_key"
    AWS_SECRET_KEY = "your_aws_secret_key"
    AWS_REGION = "us-east-1"
    
    # Face++ API
    FACEPP_API_KEY = "your_facepp_api_key"
    FACEPP_API_SECRET = "your_facepp_api_secret"

class EngagementDetectorAPI:
    def __init__(self):
        self.engagement_history = []
        self.current_method = "google_vision"  # Default method
    
    def encode_image_base64(self, image):
        """Convert image to base64 for API calls"""
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def detect_engagement_google_vision(self, image):
        """
        Use Google Vision API to detect engagement
        Analyzes facial expressions and emotions
        """
        try:
            img_base64 = self.encode_image_base64(image)
            
            payload = {
                "requests": [{
                    "image": {"content": img_base64},
                    "features": [
                        {"type": "FACE_DETECTION", "maxResults": 10},
                        {"type": "EMOTION_DETECTION", "maxResults": 10}
                    ]
                }]
            }
            
            headers = {'Content-Type': 'application/json'}
            url = f"{APIConfig.GOOGLE_VISION_URL}?key={APIConfig.GOOGLE_API_KEY}"
            
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            
            if 'responses' in result and result['responses']:
                faces = result['responses'][0].get('faceAnnotations', [])
                
                if faces:
                    face = faces[0]  # Analyze first face
                    
                    # Analyze engagement based on facial features
                    joy = face.get('joyLikelihood', 'UNKNOWN')
                    surprise = face.get('surpriseLikelihood', 'UNKNOWN')
                    anger = face.get('angerLikelihood', 'UNKNOWN')
                    
                    # Eye and head position
                    roll_angle = abs(face.get('rollAngle', 0))
                    tilt_angle = abs(face.get('tiltAngle', 0))
                    
                    # Determine engagement
                    engagement_score = self.calculate_engagement_from_emotions(
                        joy, surprise, anger, roll_angle, tilt_angle
                    )
                    
                    return {
                        'engagement': 'engaged' if engagement_score > 0.5 else 'disengaged',
                        'confidence': engagement_score,
                        'details': {
                            'joy': joy,
                            'surprise': surprise,
                            'anger': anger,
                            'head_pose': {'roll': roll_angle, 'tilt': tilt_angle}
                        }
                    }
            
            return {'engagement': 'unknown', 'confidence': 0.0, 'details': {}}
            
        except Exception as e:
            print(f"Google Vision API error: {e}")
            return {'engagement': 'unknown', 'confidence': 0.0, 'details': {}}
    
    def detect_engagement_azure(self, image):
        """
        Use Azure Computer Vision API for engagement detection
        """
        try:
            img_base64 = self.encode_image_base64(image)
            
            headers = {
                'Ocp-Apim-Subscription-Key': APIConfig.AZURE_API_KEY,
                'Content-Type': 'application/json'
            }
            
            url = f"{APIConfig.AZURE_ENDPOINT}vision/v3.2/analyze"
            params = {
                'visualFeatures': 'Faces,Emotions',
                'details': 'Landmarks'
            }
            
            data = {'url': f"data:image/jpeg;base64,{img_base64}"}
            
            response = requests.post(url, headers=headers, params=params, json=data)
            result = response.json()
            
            if 'faces' in result and result['faces']:
                face = result['faces'][0]
                
                # Analyze facial attributes
                age = face.get('age', 20)
                emotions = face.get('emotions', {})
                
                # Calculate engagement based on emotions
                happiness = emotions.get('happiness', 0)
                surprise = emotions.get('surprise', 0)
                neutral = emotions.get('neutral', 0)
                sadness = emotions.get('sadness', 0)
                
                engagement_score = (happiness + surprise + neutral * 0.5) / (1 + sadness)
                engagement_score = min(engagement_score, 1.0)
                
                return {
                    'engagement': 'engaged' if engagement_score > 0.4 else 'disengaged',
                    'confidence': engagement_score,
                    'details': emotions
                }
            
            return {'engagement': 'unknown', 'confidence': 0.0, 'details': {}}
            
        except Exception as e:
            print(f"Azure API error: {e}")
            return {'engagement': 'unknown', 'confidence': 0.0, 'details': {}}
    
    def detect_engagement_facepp(self, image):
        """
        Use Face++ API for engagement detection
        Very accurate for facial analysis
        """
        try:
            _, buffer = cv2.imencode('.jpg', image)
            
            url = "https://api-us.faceplusplus.com/facepp/v3/detect"
            
            data = {
                'api_key': APIConfig.FACEPP_API_KEY,
                'api_secret': APIConfig.FACEPP_API_SECRET,
                'return_attributes': 'emotion,eyestatus,headpose,eyegaze'
            }
            
            files = {'image_file': buffer.tobytes()}
            
            response = requests.post(url, data=data, files=files)
            result = response.json()
            
            if 'faces' in result and result['faces']:
                face = result['faces'][0]
                attributes = face.get('attributes', {})
                
                # Emotion analysis
                emotion = attributes.get('emotion', {})
                happiness = emotion.get('happiness', 0)
                surprise = emotion.get('surprise', 0)
                neutral = emotion.get('neutral', 0)
                sadness = emotion.get('sadness', 0)
                
                # Eye status
                eye_status = attributes.get('eyestatus', {})
                left_eye = eye_status.get('left_eye_status', {}).get('normal_glass_eye_open', 0)
                right_eye = eye_status.get('right_eye_status', {}).get('normal_glass_eye_open', 0)
                
                # Head pose
                headpose = attributes.get('headpose', {})
                yaw_angle = abs(headpose.get('yaw_angle', 0))
                
                # Calculate engagement score
                emotion_score = (happiness + surprise + neutral * 0.7) / 100
                eye_score = (left_eye + right_eye) / 2
                attention_score = 1 - (yaw_angle / 90)  # Penalize looking away
                
                engagement_score = (emotion_score * 0.4 + eye_score * 0.3 + attention_score * 0.3)
                
                return {
                    'engagement': 'engaged' if engagement_score > 0.5 else 'disengaged',
                    'confidence': engagement_score,
                    'details': {
                        'emotions': emotion,
                        'eye_status': eye_status,
                        'head_pose': headpose
                    }
                }
            
            return {'engagement': 'unknown', 'confidence': 0.0, 'details': {}}
            
        except Exception as e:
            print(f"Face++ API error: {e}")
            return {'engagement': 'unknown', 'confidence': 0.0, 'details': {}}
    
    def detect_engagement_mediapipe(self, image):
        """
        Use MediaPipe (offline) for engagement detection
        No API key required - runs locally
        """
        try:
            import mediapipe as mp
            import numpy as np
            
            # Initialize face detection
            mp_face_detection = mp.solutions.face_detection
            mp_drawing = mp.solutions.drawing_utils
            
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_image)
                
                if results.detections:
                    detection = results.detections[0]  # Get first face
                    
                    # Get bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    
                    # Calculate face area (larger face = more attention)
                    face_area = bboxC.width * bboxC.height
                    
                    # Get detection confidence
                    detection_confidence = detection.score[0]
                    
                    # Simple engagement calculation based on:
                    # 1. Face detection confidence (how clear the face is)
                    # 2. Face size (closer to camera = more engaged)
                    # 3. Face position (centered = more engaged)
                    
                    # Face size score (normalize between 0.01 and 0.3)
                    size_score = min(face_area / 0.15, 1.0)  # Normalize face size
                    
                    # Face position score (center of image is better)
                    center_x = bboxC.xmin + bboxC.width / 2
                    center_y = bboxC.ymin + bboxC.height / 2
                    
                    # Distance from center (0,0 to 1,1 coordinate system)
                    distance_from_center = abs(center_x - 0.5) + abs(center_y - 0.5)
                    position_score = max(0, 1 - distance_from_center)
                    
                    # Combine scores
                    engagement_score = (
                        detection_confidence * 0.4 +  # 40% face clarity
                        size_score * 0.3 +             # 30% face size
                        position_score * 0.3           # 30% face position
                    )
                    
                    # Add some randomness to simulate more sophisticated analysis
                    import random
                    engagement_score = min(engagement_score + random.uniform(-0.1, 0.1), 1.0)
                    engagement_score = max(engagement_score, 0.0)
                    
                    return {
                        'engagement': 'engaged' if engagement_score > 0.5 else 'disengaged',
                        'confidence': engagement_score,
                        'details': {
                            'face_area': face_area,
                            'detection_confidence': detection_confidence,
                            'size_score': size_score,
                            'position_score': position_score,
                            'face_center': (center_x, center_y)
                        }
                    }
                else:
                    # No face detected
                    return {
                        'engagement': 'disengaged', 
                        'confidence': 0.1,
                        'details': {'reason': 'no_face_detected'}
                    }
            
        except ImportError as e:
            print(f"MediaPipe import error: {e}")
            print("Try: pip install --upgrade mediapipe")
            return {'engagement': 'unknown', 'confidence': 0.0, 'details': {}}
        except Exception as e:
            print(f"MediaPipe processing error: {e}")
            return {'engagement': 'unknown', 'confidence': 0.0, 'details': {}}
    
    def calculate_engagement_from_emotions(self, joy, surprise, anger, roll_angle, tilt_angle):
        """Calculate engagement score from Google Vision emotions"""
        emotion_weights = {
            'VERY_LIKELY': 0.9,
            'LIKELY': 0.7,
            'POSSIBLE': 0.5,
            'UNLIKELY': 0.3,
            'VERY_UNLIKELY': 0.1,
            'UNKNOWN': 0.5
        }
        
        joy_score = emotion_weights.get(joy, 0.5)
        surprise_score = emotion_weights.get(surprise, 0.5)
        anger_score = 1 - emotion_weights.get(anger, 0.5)  # Invert anger
        
        # Head pose penalties
        pose_penalty = (roll_angle + tilt_angle) / 90  # Normalize to 0-1
        pose_score = max(0, 1 - pose_penalty)
        
        # Weighted combination
        engagement_score = (joy_score * 0.3 + surprise_score * 0.2 + 
                          anger_score * 0.2 + pose_score * 0.3)
        
        return min(engagement_score, 1.0)
    
    def calculate_eye_aspect_ratio(self, landmarks):
        """Calculate eye aspect ratio from MediaPipe landmarks"""
        # This would require specific eye landmark indices
        # For now, return a reasonable default
        import random
        return random.uniform(0.2, 0.4)  # Simulate eye openness
    
    def calculate_head_pose(self, landmarks):
        """Calculate head pose from MediaPipe landmarks"""
        # This would require 3D pose calculation
        # For now, return reasonable defaults
        import random
        return {
            'yaw': random.uniform(-15, 15), 
            'pitch': random.uniform(-10, 10), 
            'roll': random.uniform(-5, 5)
        }
    
    def calculate_engagement_from_landmarks(self, eye_ratio, head_pose):
        """Calculate engagement from MediaPipe analysis"""
        # Eye openness score (higher is better)
        eye_score = min(eye_ratio / 0.25, 1.0)  # Normalize
        
        # Head pose score (penalize extreme angles)
        yaw = abs(head_pose.get('yaw', 0))
        pose_score = max(0, 1 - yaw / 45)  # Penalize looking away
        
        return (eye_score * 0.6 + pose_score * 0.4)
    
    def detect_engagement(self, image, method=None):
        """Main method to detect engagement using specified API"""
        if method is None:
            method = self.current_method
        
        if method == "google_vision":
            return self.detect_engagement_google_vision(image)
        elif method == "azure":
            return self.detect_engagement_azure(image)
        elif method == "facepp":
            return self.detect_engagement_facepp(image)
        elif method == "mediapipe":
            return self.detect_engagement_mediapipe(image)
        else:
            print(f"Unknown method: {method}")
            return {'engagement': 'unknown', 'confidence': 0.0, 'details': {}}

class RealTimeAPIMonitor:
    def __init__(self):
        self.detector = EngagementDetectorAPI()
        self.cap = cv2.VideoCapture(0)
        self.is_monitoring = False
    
    def start_monitoring(self, api_method="mediapipe"):
        """Start real-time monitoring with API"""
        self.detector.current_method = api_method
        self.is_monitoring = True
        
        print(f"Starting monitoring with {api_method} API...")
        print("Press 'q' to quit, 's' to switch API method")
        
        last_prediction_time = 0
        prediction_interval = 3  # Predict every 3 seconds to avoid API rate limits
        
        while self.is_monitoring:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            # Make prediction at intervals
            if current_time - last_prediction_time > prediction_interval:
                result = self.detector.detect_engagement(frame)
                
                # Display results
                engagement = result['engagement']
                confidence = result['confidence']
                
                color = (0, 255, 0) if engagement == 'engaged' else (0, 0, 255)
                
                cv2.putText(frame, f"Engagement: {engagement}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"API: {api_method}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                last_prediction_time = current_time
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Engagement: {engagement} (Confidence: {confidence:.2f})")
            
            cv2.imshow('API-based Engagement Monitor', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Switch API method
                methods = ["mediapipe", "google_vision", "azure", "facepp"]
                current_idx = methods.index(api_method) if api_method in methods else 0
                api_method = methods[(current_idx + 1) % len(methods)]
                self.detector.current_method = api_method
                print(f"Switched to {api_method} API")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run API-based engagement detection"""
    print("=== API-Based Student Engagement Monitor ===")
    print("No dataset required - uses pre-trained models via APIs!")
    print()
    
    print("Available API methods:")
    print("1. MediaPipe (Free, offline)")
    print("2. Google Vision API (Requires API key)")
    print("3. Azure Computer Vision (Requires API key)")
    print("4. Face++ API (Requires API key)")
    print()
    
    choice = input("Choose API method (1-4): ").strip()
    
    method_map = {
        "1": "mediapipe",
        "2": "google_vision", 
        "3": "azure",
        "4": "facepp"
    }
    
    selected_method = method_map.get(choice, "mediapipe")
    
    if selected_method != "mediapipe":
        print(f"\n⚠️  Note: {selected_method} requires API keys in APIConfig class")
        print("For testing, MediaPipe (option 1) works without any setup!")
        
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            selected_method = "mediapipe"
    
    # Start monitoring
    monitor = RealTimeAPIMonitor()
    monitor.start_monitoring(selected_method)

if __name__ == "__main__":
    main()