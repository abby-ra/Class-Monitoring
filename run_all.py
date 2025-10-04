#!/usr/bin/env python3
"""
🎓 STUDENT ENGAGEMENT MONITORING SYSTEM
Integrated launcher - runs the complete system
"""

import os
import sys
import subprocess
from datetime import datetime

print("=" * 60)
print("🎓 STUDENT ENGAGEMENT MONITORING SYSTEM")
print("=" * 60)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🚀 Launching INTEGRATED monitoring system...")
print("=" * 60)

# Check if integrated app exists
if not os.path.exists('student_monitor.py'):
    print("❌ Error: student_monitor.py not found!")
    print("💡 Make sure all files are in the same directory")
    input("Press Enter to exit...")
    sys.exit(1)

print("✅ Found integrated monitoring system")
print("🎖️ Features: ML + OpenCV + Smart Analytics + Professional GUI")
print("⏱️ Starting in 2 seconds...")
print("-" * 60)

import time
time.sleep(2)

try:
    # Launch the integrated system
    subprocess.run([sys.executable, 'student_monitor.py'])
    print("\n✅ Monitoring session completed!")
    
except KeyboardInterrupt:
    print("\n⏹️ Monitoring interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    print("💡 Please check your camera and dependencies")

print("\n👋 Thanks for using Student Engagement Monitor!")
input("Press Enter to exit...")