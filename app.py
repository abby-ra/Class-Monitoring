"""
Main application for Student Engagement Monitoring System
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils.data_preprocessing import DataPreprocessor
from models.cnn_model import EngagementCNN
from real_time.webcam_monitor import RealTimeMonitor
from real_time.alert_system import AlertSystem
from utils.evaluation import ModelEvaluator

class StudentEngagementApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Student Engagement Monitoring System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Initialize components
        self.model = None
        self.monitor = None
        self.alert_system = AlertSystem()
        self.is_monitoring = False
        
        # Create UI
        self.create_ui()
        
        # Check if model exists
        self.check_model_availability()
    
    def create_ui(self):
        """Create the user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_monitoring_tab()
        self.create_training_tab()
        self.create_analytics_tab()
        self.create_settings_tab()
    
    def create_monitoring_tab(self):
        """Create real-time monitoring tab"""
        self.monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.monitoring_frame, text="Real-time Monitoring")
        
        # Control panel
        control_frame = ttk.LabelFrame(self.monitoring_frame, text="Controls")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Start/Stop buttons
        self.start_btn = ttk.Button(control_frame, text="Start Monitoring", 
                                   command=self.start_monitoring, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring", 
                                  command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Model not loaded")
        self.status_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Current prediction frame
        prediction_frame = ttk.LabelFrame(self.monitoring_frame, text="Current Prediction")
        prediction_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.engagement_label = ttk.Label(prediction_frame, text="Engagement: Unknown", 
                                         font=("Arial", 16, "bold"))
        self.engagement_label.pack(pady=10)
        
        self.confidence_label = ttk.Label(prediction_frame, text="Confidence: 0.00", 
                                         font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        # Recent alerts frame
        alerts_frame = ttk.LabelFrame(self.monitoring_frame, text="Recent Alerts")
        alerts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for alerts
        self.alerts_tree = ttk.Treeview(alerts_frame, columns=("Time", "Type", "Message"), 
                                       show="tree headings")
        self.alerts_tree.heading("#0", text="ID")
        self.alerts_tree.heading("Time", text="Time")
        self.alerts_tree.heading("Type", text="Alert Type")
        self.alerts_tree.heading("Message", text="Message")
        
        # Configure column widths
        self.alerts_tree.column("#0", width=50)
        self.alerts_tree.column("Time", width=100)
        self.alerts_tree.column("Type", width=150)
        self.alerts_tree.column("Message", width=400)
        
        self.alerts_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Clear alerts button
        clear_alerts_btn = ttk.Button(alerts_frame, text="Clear Alerts", 
                                     command=self.clear_alerts)
        clear_alerts_btn.pack(pady=5)
    
    def create_training_tab(self):
        """Create model training tab"""
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Model Training")
        
        # Dataset configuration
        dataset_frame = ttk.LabelFrame(self.training_frame, text="Dataset Configuration")
        dataset_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Dataset path
        ttk.Label(dataset_frame, text="Dataset Path:").pack(anchor=tk.W, padx=5, pady=2)
        self.dataset_path_var = tk.StringVar(value=DATASET_PATH)
        dataset_entry_frame = ttk.Frame(dataset_frame)
        dataset_entry_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.dataset_entry = ttk.Entry(dataset_entry_frame, textvariable=self.dataset_path_var)
        self.dataset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(dataset_entry_frame, text="Browse", 
                               command=self.browse_dataset)
        browse_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Training parameters
        params_frame = ttk.LabelFrame(self.training_frame, text="Training Parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.epochs_var = tk.StringVar(value=str(EPOCHS))
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.batch_size_var = tk.StringVar(value=str(BATCH_SIZE))
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.lr_var = tk.StringVar(value=str(LEARNING_RATE))
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Use augmentation
        self.use_augmentation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Use Data Augmentation", 
                       variable=self.use_augmentation_var).grid(row=1, column=2, columnspan=2, 
                                                               sticky=tk.W, padx=5, pady=2)
        
        # Training control
        training_control_frame = ttk.Frame(self.training_frame)
        training_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.train_btn = ttk.Button(training_control_frame, text="Start Training", 
                                   command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_train_btn = ttk.Button(training_control_frame, text="Stop Training", 
                                        command=self.stop_training, state=tk.DISABLED)
        self.stop_train_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.training_progress = ttk.Progressbar(training_control_frame, mode='indeterminate')
        self.training_progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Training log
        log_frame = ttk.LabelFrame(self.training_frame, text="Training Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.training_log = tk.Text(log_frame, height=15, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.training_log.yview)
        self.training_log.configure(yscrollcommand=log_scrollbar.set)
        
        self.training_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def create_analytics_tab(self):
        """Create analytics and reporting tab"""
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text="Analytics")
        
        # Session statistics
        stats_frame = ttk.LabelFrame(self.analytics_frame, text="Session Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create statistics labels
        self.stats_labels = {}
        stats_info = [
            ("Total Readings", "total_readings"),
            ("Engagement Rate", "engagement_rate"),
            ("Average Confidence", "avg_confidence"),
            ("Session Duration", "duration"),
            ("Total Alerts", "total_alerts")
        ]
        
        for i, (label_text, key) in enumerate(stats_info):
            row = i // 3
            col = (i % 3) * 2
            
            ttk.Label(stats_frame, text=f"{label_text}:").grid(row=row, column=col, 
                                                              sticky=tk.W, padx=5, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="0", font=("Arial", 10, "bold"))
            self.stats_labels[key].grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=2)
        
        # Refresh button
        refresh_btn = ttk.Button(stats_frame, text="Refresh Statistics", 
                                command=self.refresh_statistics)
        refresh_btn.grid(row=2, column=4, padx=5, pady=2)
        
        # Chart frame
        chart_frame = ttk.LabelFrame(self.analytics_frame, text="Engagement Chart")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Export buttons
        export_frame = ttk.Frame(self.analytics_frame)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(export_frame, text="Export Session Report", 
                  command=self.export_session_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export Chart", 
                  command=self.export_chart).pack(side=tk.LEFT, padx=5)
    
    def create_settings_tab(self):
        """Create settings tab"""
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        
        # Model settings
        model_frame = ttk.LabelFrame(self.settings_frame, text="Model Settings")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Confidence threshold
        ttk.Label(model_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.confidence_threshold_var = tk.DoubleVar(value=CONFIDENCE_THRESHOLD)
        confidence_scale = ttk.Scale(model_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_threshold_var, 
                                   orient=tk.HORIZONTAL, length=200)
        confidence_scale.grid(row=0, column=1, padx=5, pady=2)
        
        self.confidence_value_label = ttk.Label(model_frame, text=f"{CONFIDENCE_THRESHOLD:.2f}")
        self.confidence_value_label.grid(row=0, column=2, padx=5, pady=2)
        
        # Update confidence value display
        confidence_scale.configure(command=self.update_confidence_display)
        
        # Alert settings
        alert_frame = ttk.LabelFrame(self.settings_frame, text="Alert Settings")
        alert_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Enable/disable alerts
        self.enable_alerts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(alert_frame, text="Enable Alerts", 
                       variable=self.enable_alerts_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Alert sound
        self.alert_sound_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(alert_frame, text="Alert Sound", 
                       variable=self.alert_sound_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Save settings button
        save_settings_btn = ttk.Button(self.settings_frame, text="Save Settings", 
                                      command=self.save_settings)
        save_settings_btn.pack(pady=10)
        
        # About section
        about_frame = ttk.LabelFrame(self.settings_frame, text="About")
        about_frame.pack(fill=tk.X, padx=10, pady=5)
        
        about_text = """
Student Engagement Monitoring System v1.0

This application uses deep learning to monitor student engagement
in real-time through webcam analysis. It provides alerts and
analytics to help improve classroom attention and learning outcomes.

Features:
• Real-time engagement detection
• Alert system for low engagement
• Training custom models
• Analytics and reporting
• Configurable settings
        """
        
        ttk.Label(about_frame, text=about_text, justify=tk.LEFT).pack(padx=10, pady=10)
    
    def check_model_availability(self):
        """Check if trained model is available"""
        if os.path.exists(MODEL_SAVE_PATH):
            try:
                self.model = EngagementCNN()
                self.model.load_model(MODEL_SAVE_PATH)
                self.monitor = RealTimeMonitor()
                self.start_btn.configure(state=tk.NORMAL)
                self.status_label.configure(text="Model loaded - Ready to monitor")
            except Exception as e:
                self.status_label.configure(text=f"Error loading model: {str(e)}")
        else:
            self.status_label.configure(text="No trained model found - Please train a model first")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.model is None:
            messagebox.showerror("Error", "No model loaded. Please train a model first.")
            return
        
        self.is_monitoring = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_label.configure(text="Monitoring active...")
        
        # Start monitoring in separate thread
        self.monitoring_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_label.configure(text="Monitoring stopped")
        
        # Show session summary
        self.alert_system.print_session_summary()
        self.refresh_statistics()
    
    def monitor_loop(self):
        """Main monitoring loop"""
        try:
            # This would be replaced with actual webcam monitoring
            import time
            import random
            
            while self.is_monitoring:
                # Simulate prediction (replace with actual webcam prediction)
                engagement = random.choice(['engaged', 'disengaged'])
                confidence = random.uniform(0.4, 0.95)
                
                # Update UI in main thread
                self.root.after(0, self.update_prediction_display, engagement, confidence)
                
                # Log to alert system
                self.alert_system.log_engagement(engagement, confidence)
                
                # Check for new alerts
                recent_alerts = self.alert_system.get_recent_alerts(minutes=1)
                if recent_alerts:
                    self.root.after(0, self.update_alerts_display)
                
                time.sleep(2)  # Update every 2 seconds
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Monitoring Error", str(e)))
    
    def update_prediction_display(self, engagement, confidence):
        """Update prediction display"""
        color = "green" if engagement == "engaged" else "red"
        self.engagement_label.configure(text=f"Engagement: {engagement.title()}", 
                                       foreground=color)
        self.confidence_label.configure(text=f"Confidence: {confidence:.2f}")
    
    def update_alerts_display(self):
        """Update alerts display"""
        # Clear existing items
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
        
        # Add recent alerts
        recent_alerts = self.alert_system.get_recent_alerts(minutes=60)  # Last hour
        for i, alert in enumerate(recent_alerts[-20:]):  # Show last 20 alerts
            timestamp = datetime.fromisoformat(alert['timestamp'])
            time_str = timestamp.strftime('%H:%M:%S')
            
            self.alerts_tree.insert('', 'end', text=str(i+1), 
                                   values=(time_str, alert['type'], alert['message']))
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alert_system.clear_alerts()
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
        self.refresh_statistics()
    
    def refresh_statistics(self):
        """Refresh statistics display"""
        summary = self.alert_system.get_session_summary()
        
        if summary:
            self.stats_labels['total_readings'].configure(text=str(summary['total_readings']))
            self.stats_labels['engagement_rate'].configure(text=f"{summary['engagement_percentage']:.1f}%")
            self.stats_labels['avg_confidence'].configure(text=f"{summary['average_confidence']:.2f}")
            self.stats_labels['duration'].configure(text=f"{summary['session_duration_seconds']:.0f}s")
            self.stats_labels['total_alerts'].configure(text=str(summary['total_alerts']))
            
            # Update chart
            self.update_engagement_chart()
        else:
            # Reset to zeros
            for key in self.stats_labels:
                self.stats_labels[key].configure(text="0")
    
    def update_engagement_chart(self):
        """Update engagement chart"""
        if not self.alert_system.engagement_history:
            return
        
        # Prepare data
        timestamps = []
        engagements = []
        
        for reading in self.alert_system.engagement_history[-50:]:  # Last 50 readings
            timestamp = datetime.fromisoformat(reading['timestamp'])
            timestamps.append(timestamp)
            engagements.append(1 if reading['engagement'] == 'engaged' else 0)
        
        # Clear and plot
        self.ax.clear()
        self.ax.plot(timestamps, engagements, 'bo-', linewidth=2, markersize=4)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Engagement (1=Engaged, 0=Disengaged)')
        self.ax.set_title('Student Engagement Over Time')
        self.ax.grid(True, alpha=0.3)
        
        # Format x-axis
        self.fig.autofmt_xdate()
        
        # Refresh canvas
        self.canvas.draw()
    
    def browse_dataset(self):
        """Browse for dataset directory"""
        directory = filedialog.askdirectory(title="Select Dataset Directory")
        if directory:
            self.dataset_path_var.set(directory)
    
    def start_training(self):
        """Start model training"""
        # Validate parameters
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            learning_rate = float(self.lr_var.get())
            dataset_path = self.dataset_path_var.get()
            
            if not os.path.exists(dataset_path):
                messagebox.showerror("Error", "Dataset path does not exist!")
                return
                
        except ValueError:
            messagebox.showerror("Error", "Invalid training parameters!")
            return
        
        # Update UI
        self.train_btn.configure(state=tk.DISABLED)
        self.stop_train_btn.configure(state=tk.NORMAL)
        self.training_progress.start()
        
        # Clear log
        self.training_log.delete(1.0, tk.END)
        
        # Start training in separate thread
        self.training_thread = threading.Thread(target=self.train_model_thread, 
                                               args=(epochs, batch_size, learning_rate, dataset_path),
                                               daemon=True)
        self.training_thread.start()
    
    def train_model_thread(self, epochs, batch_size, learning_rate, dataset_path):
        """Train model in separate thread"""
        try:
            self.log_training("Starting model training...")
            
            # Initialize model
            model = EngagementCNN()
            
            # Load and preprocess data
            self.log_training("Loading dataset...")
            preprocessor = DataPreprocessor()
            
            # This would be replaced with actual training logic
            import time
            for epoch in range(epochs):
                self.log_training(f"Epoch {epoch+1}/{epochs}")
                time.sleep(1)  # Simulate training time
                
                if not hasattr(self, 'training_thread') or not self.training_thread.is_alive():
                    break
            
            self.log_training("Training completed successfully!")
            self.log_training(f"Model saved to: {MODEL_SAVE_PATH}")
            
            # Update model availability
            self.root.after(0, self.check_model_availability)
            
        except Exception as e:
            self.log_training(f"Training error: {str(e)}")
        finally:
            self.root.after(0, self.training_finished)
    
    def stop_training(self):
        """Stop model training"""
        self.training_thread = None
        self.training_finished()
        self.log_training("Training stopped by user")
    
    def training_finished(self):
        """Called when training is finished"""
        self.train_btn.configure(state=tk.NORMAL)
        self.stop_train_btn.configure(state=tk.DISABLED)
        self.training_progress.stop()
    
    def log_training(self, message):
        """Log training message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        
        def update_log():
            self.training_log.insert(tk.END, log_message)
            self.training_log.see(tk.END)
        
        self.root.after(0, update_log)
    
    def update_confidence_display(self, value):
        """Update confidence threshold display"""
        self.confidence_value_label.configure(text=f"{float(value):.2f}")
    
    def save_settings(self):
        """Save application settings"""
        settings = {
            'confidence_threshold': self.confidence_threshold_var.get(),
            'enable_alerts': self.enable_alerts_var.get(),
            'alert_sound': self.alert_sound_var.get()
        }
        
        settings_file = os.path.join(LOGS_DIR, 'app_settings.json')
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        
        messagebox.showinfo("Settings", "Settings saved successfully!")
    
    def export_session_report(self):
        """Export session report"""
        summary = self.alert_system.get_session_summary()
        
        if not summary:
            messagebox.showwarning("Export", "No session data to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Session Report"
        )
        
        if filename:
            # Add additional info
            report = {
                'export_timestamp': datetime.now().isoformat(),
                'session_summary': summary,
                'engagement_history': self.alert_system.engagement_history,
                'alerts': self.alert_system.alerts_log
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            messagebox.showinfo("Export", f"Session report exported to:\n{filename}")
    
    def export_chart(self):
        """Export engagement chart"""
        if not self.alert_system.engagement_history:
            messagebox.showwarning("Export", "No data to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Export Chart"
        )
        
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Export", f"Chart exported to:\n{filename}")
    
    def run(self):
        """Run the application"""
        # Create logs directory if it doesn't exist
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        # Start the GUI
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting Student Engagement Monitoring System...")
    
    # Create and run the application
    app = StudentEngagementApp()
    app.run()

if __name__ == "__main__":
    main()