"""
Alert system for student engagement monitoring
"""

import time
import threading
from datetime import datetime
import json
import os
from config import *

class AlertSystem:
    def __init__(self):
        self.alerts_log = []
        self.is_monitoring = False
        self.alert_thread = None
        self.current_engagement = "unknown"
        self.engagement_history = []
        self.low_engagement_start = None
        
    def log_engagement(self, engagement, confidence):
        """Log engagement data"""
        timestamp = datetime.now()
        
        engagement_data = {
            'timestamp': timestamp.isoformat(),
            'engagement': engagement,
            'confidence': confidence
        }
        
        self.engagement_history.append(engagement_data)
        self.current_engagement = engagement
        
        # Check for alert conditions
        self.check_alert_conditions(engagement, confidence, timestamp)
    
    def check_alert_conditions(self, engagement, confidence, timestamp):
        """Check if alert conditions are met"""
        # Low engagement alert
        if engagement == 'disengaged' and confidence > CONFIDENCE_THRESHOLD:
            if self.low_engagement_start is None:
                self.low_engagement_start = timestamp
            else:
                # Check if low engagement has lasted too long
                duration = (timestamp - self.low_engagement_start).total_seconds()
                if duration > 30:  # 30 seconds of disengagement
                    self.trigger_alert("LOW_ENGAGEMENT", 
                                     f"Student has been disengaged for {duration:.0f} seconds", 
                                     timestamp)
        else:
            self.low_engagement_start = None
        
        # Low confidence alert
        if confidence < 0.5:
            self.trigger_alert("LOW_CONFIDENCE", 
                             f"Low prediction confidence: {confidence:.2f}", 
                             timestamp)
    
    def trigger_alert(self, alert_type, message, timestamp):
        """Trigger an alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp.isoformat(),
            'engagement_level': self.current_engagement
        }
        
        self.alerts_log.append(alert)
        
        # Print alert to console
        print(f"\n🚨 ALERT [{alert_type}]: {message}")
        print(f"   Time: {timestamp.strftime('%H:%M:%S')}")
        
        # Save to file
        self.save_alert_to_file(alert)
    
    def save_alert_to_file(self, alert):
        """Save alert to log file"""
        log_file = os.path.join(LOGS_DIR, 'alerts.json')
        
        # Load existing alerts
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    alerts = json.load(f)
                except json.JSONDecodeError:
                    alerts = []
        else:
            alerts = []
        
        # Add new alert
        alerts.append(alert)
        
        # Save back to file
        with open(log_file, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def get_session_summary(self):
        """Get summary of the monitoring session"""
        if not self.engagement_history:
            return {}
        
        total_readings = len(self.engagement_history)
        engaged_count = sum(1 for reading in self.engagement_history 
                           if reading['engagement'] == 'engaged')
        disengaged_count = total_readings - engaged_count
        
        # Calculate average confidence
        avg_confidence = sum(reading['confidence'] for reading in self.engagement_history) / total_readings
        
        # Get session duration
        if total_readings > 1:
            start_time = datetime.fromisoformat(self.engagement_history[0]['timestamp'])
            end_time = datetime.fromisoformat(self.engagement_history[-1]['timestamp'])
            duration = (end_time - start_time).total_seconds()
        else:
            duration = 0
        
        summary = {
            'session_duration_seconds': duration,
            'total_readings': total_readings,
            'engaged_count': engaged_count,
            'disengaged_count': disengaged_count,
            'engagement_percentage': (engaged_count / total_readings) * 100 if total_readings > 0 else 0,
            'average_confidence': avg_confidence,
            'total_alerts': len(self.alerts_log),
            'alerts_by_type': {}
        }
        
        # Count alerts by type
        for alert in self.alerts_log:
            alert_type = alert['type']
            summary['alerts_by_type'][alert_type] = summary['alerts_by_type'].get(alert_type, 0) + 1
        
        return summary
    
    def print_session_summary(self):
        """Print session summary"""
        summary = self.get_session_summary()
        
        if not summary:
            print("No session data available")
            return
        
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Duration: {summary['session_duration_seconds']:.0f} seconds")
        print(f"Total readings: {summary['total_readings']}")
        print(f"Engagement: {summary['engagement_percentage']:.1f}% ({summary['engaged_count']}/{summary['total_readings']})")
        print(f"Average confidence: {summary['average_confidence']:.2f}")
        print(f"Total alerts: {summary['total_alerts']}")
        
        if summary['alerts_by_type']:
            print("\nAlerts by type:")
            for alert_type, count in summary['alerts_by_type'].items():
                print(f"  {alert_type}: {count}")
        
        print("="*50)
    
    def save_session_summary(self):
        """Save session summary to file"""
        summary = self.get_session_summary()
        
        if not summary:
            return
        
        # Add timestamp to summary
        summary['session_end_time'] = datetime.now().isoformat()
        
        # Save to file
        summary_file = os.path.join(LOGS_DIR, f"session_summary_{int(time.time())}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Session summary saved to: {summary_file}")
    
    def get_recent_alerts(self, minutes=10):
        """Get alerts from the last N minutes"""
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        
        recent_alerts = []
        for alert in self.alerts_log:
            alert_time = datetime.fromisoformat(alert['timestamp']).timestamp()
            if alert_time > cutoff_time:
                recent_alerts.append(alert)
        
        return recent_alerts
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts_log.clear()
        self.engagement_history.clear()
        print("All alerts and engagement history cleared")

def main():
    """Test alert system"""
    alert_system = AlertSystem()
    
    # Simulate some engagement data
    import random
    
    print("Testing alert system...")
    
    for i in range(10):
        engagement = random.choice(['engaged', 'disengaged'])
        confidence = random.uniform(0.3, 0.9)
        
        alert_system.log_engagement(engagement, confidence)
        time.sleep(0.5)  # Short delay to simulate real-time
    
    # Print summary
    alert_system.print_session_summary()
    
    print("\nAlert system test completed!")

if __name__ == "__main__":
    main()