"""
Quick Test for API-based Engagement Detection
This demonstrates how to use APIs without any custom dataset
"""

import cv2
import time
from api_based_detector import EngagementDetectorAPI, RealTimeAPIMonitor

def quick_test():
    """Quick test of API-based detection"""
    print("🚀 API-Based Student Engagement Detection")
    print("=" * 50)
    print("✅ No dataset required!")
    print("✅ No model training needed!")
    print("✅ Uses pre-trained AI models via APIs")
    print("=" * 50)
    
    # Test with MediaPipe (works offline, no API key needed)
    print("\n📹 Testing with MediaPipe (Free & Offline)...")
    
    try:
        # Initialize detector
        detector = EngagementDetectorAPI()
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam")
            return
        
        print("✅ Webcam opened successfully")
        print("📊 Starting engagement detection...")
        print("💡 Look at the camera and try different expressions!")
        print("⌨️ Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every 30 frames (about 1 second at 30 FPS)
            if frame_count % 30 == 0:
                print("🔍 Analyzing frame...")
                
                # Detect engagement using MediaPipe
                result = detector.detect_engagement(frame, method="mediapipe")
                
                engagement = result['engagement']
                confidence = result['confidence']
                
                # Display results
                status_emoji = "😊" if engagement == "engaged" else "😴"
                print(f"{status_emoji} Engagement: {engagement.upper()} (Confidence: {confidence:.2f})")
                
                # Show on frame
                color = (0, 255, 0) if engagement == 'engaged' else (0, 0, 255)
                cv2.putText(frame, f"Status: {engagement}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('API-based Engagement Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")

def show_api_options():
    """Show available API options"""
    print("\n🔧 Available API Options:")
    print("=" * 40)
    
    apis = [
        {
            "name": "MediaPipe",
            "cost": "FREE",
            "setup": "No API key needed",
            "accuracy": "Good",
            "speed": "Fast",
            "offline": "Yes"
        },
        {
            "name": "Google Vision API",
            "cost": "$1.50/1000 requests",
            "setup": "Google Cloud API key",
            "accuracy": "Excellent",
            "speed": "Medium",
            "offline": "No"
        },
        {
            "name": "Azure Computer Vision",
            "cost": "$1.00/1000 requests",
            "setup": "Azure subscription",
            "accuracy": "Excellent",
            "speed": "Medium",
            "offline": "No"
        },
        {
            "name": "Face++ API",
            "cost": "Free tier available",
            "setup": "Face++ account",
            "accuracy": "Very Good",
            "speed": "Fast",
            "offline": "No"
        }
    ]
    
    for api in apis:
        print(f"\n📡 {api['name']}")
        print(f"   💰 Cost: {api['cost']}")
        print(f"   🔑 Setup: {api['setup']}")
        print(f"   🎯 Accuracy: {api['accuracy']}")
        print(f"   ⚡ Speed: {api['speed']}")
        print(f"   🔒 Offline: {api['offline']}")

def setup_instructions():
    """Show setup instructions for different APIs"""
    print("\n📚 Setup Instructions:")
    print("=" * 30)
    
    print("\n1️⃣ MediaPipe (Recommended for beginners):")
    print("   • Already included in requirements.txt")
    print("   • Works immediately after installation")
    print("   • pip install mediapipe")
    
    print("\n2️⃣ Google Vision API:")
    print("   • Go to Google Cloud Console")
    print("   • Enable Vision API")
    print("   • Create API key")
    print("   • Add key to APIConfig.GOOGLE_API_KEY")
    
    print("\n3️⃣ Azure Computer Vision:")
    print("   • Create Azure account")
    print("   • Create Computer Vision resource")
    print("   • Get endpoint and API key")
    print("   • Add to APIConfig.AZURE_* variables")
    
    print("\n4️⃣ Face++ API:")
    print("   • Register at https://www.faceplusplus.com/")
    print("   • Get API key and secret") 
    print("   • Add to APIConfig.FACEPP_* variables")

def main():
    """Main function"""
    print("🎓 Student Engagement Monitoring - API Version")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. 🚀 Quick Test (MediaPipe)")
        print("2. 📡 View API Options")
        print("3. 📚 Setup Instructions")
        print("4. 🎬 Full Real-time Monitor")
        print("5. ❌ Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            quick_test()
        elif choice == '2':
            show_api_options()
        elif choice == '3':
            setup_instructions()
        elif choice == '4':
            print("\n🎬 Starting full real-time monitor...")
            monitor = RealTimeAPIMonitor()
            monitor.start_monitoring("mediapipe")
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice! Please try again.")

if __name__ == "__main__":
    main()