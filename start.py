#!/usr/bin/env python3
"""
🎓 STUDENT ENGAGEMENT MONITORING SYSTEM
The BEST and ONLY option you need! No menus, no choices - just works!
"""

import os
import sys
import subprocess
from datetime import datetime

print("=" * 70)
print("🎓 INTEGRATED STUDENT ENGAGEMENT MONITORING SYSTEM")
print("=" * 70)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🚀 Launching INTEGRATED monitoring system...")
print("🔗 All components connected: ML + OpenCV + Utils + Real-time")
print("=" * 70)

# Check if the best app exists
if not os.path.exists('student_monitor.py'):
    print("❌ Error: student_monitor.py not found!")
    print("💡 Make sure all files are in the same directory")
    input("Press Enter to exit...")
    sys.exit(1)

print("✅ Found integrated monitoring system")
print("🎖️ Features: ML Models + OpenCV + Smart Alerts + Real-time Analytics")
print("🔗 Connected: Configuration + Utilities + Professional GUI")
print("⏱️ Starting integrated system now...")
print("-" * 70)

try:
    # Launch the BEST option directly
    subprocess.run([sys.executable, 'student_monitor.py'])
    print("\n✅ Monitoring session completed!")
    
except KeyboardInterrupt:
    print("\n⏹️ Monitoring interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {str(e)}")

print("\n👋 Thanks for using Student Engagement Monitor!")
input("Press Enter to exit...")
