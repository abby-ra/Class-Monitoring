# Student Engagement Monitoring System

A comprehensive deep learning application for real-time student engagement detection using computer vision and machine learning techniques.

## 🎯 Features

- **Real-time Monitoring**: Live webcam-based engagement detection
- **Deep Learning Model**: Custom CNN architecture for engagement classification
- **Alert System**: Automated alerts for low engagement periods
- **GUI Application**: User-friendly interface with multiple tabs
- **Analytics Dashboard**: Comprehensive reporting and visualization
- **Model Training**: Built-in training pipeline with data augmentation
- **Configurable Settings**: Customizable thresholds and parameters

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Webcam (for real-time monitoring)
- Windows/Linux/macOS
- Minimum 4GB RAM
- GPU recommended for training (optional)

### Dependencies
All required packages are listed in `requirements.txt`. Key dependencies include:
- TensorFlow 2.13.0
- OpenCV 4.8.1.78
- Scikit-learn 1.3.0
- Matplotlib
- Tkinter
- NumPy, Pandas

## 🚀 Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd Class_Monitoring
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p data/processed models/saved_models logs
   ```

## 📁 Project Structure

```
Class_Monitoring/
├── app.py                          # Main GUI application
├── config.py                       # Configuration settings
├── requirements.txt                 # Python dependencies
├── train_model.py                   # Model training script
├── main.py                         # Original script (legacy)
├── data/                           # Data directory
│   └── processed/                  # Processed datasets
├── models/                         # Model definitions
│   ├── cnn_model.py               # CNN architecture
│   └── saved_models/              # Trained models
├── utils/                          # Utility modules
│   ├── data_preprocessing.py      # Data preprocessing
│   ├── evaluation.py              # Model evaluation
│   └── data_augmentation.py       # Data augmentation
├── real_time/                      # Real-time monitoring
│   ├── webcam_monitor.py          # Webcam integration
│   └── alert_system.py            # Alert management
└── logs/                           # Application logs
```

## 🗂️ Dataset Setup

1. **Prepare your dataset** with the following structure:
   ```
   dataset/
   ├── engaged/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── disengaged/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

2. **Update dataset path** in `config.py`:
   ```python
   DATASET_PATH = "path/to/your/dataset"
   ```

## 🎮 Usage

### 1. GUI Application (Recommended)
Launch the main application with graphical interface:
```bash
python app.py
```

The GUI provides four main tabs:
- **Real-time Monitoring**: Live engagement detection
- **Model Training**: Train custom models
- **Analytics**: View statistics and charts
- **Settings**: Configure application parameters

### 2. Command Line Training
Train a model using the command line:
```bash
python train_model.py
```

### 3. Real-time Monitoring Only
Run only the webcam monitoring:
```bash
python real_time/webcam_monitor.py
```

## 📊 Model Training

### Using GUI
1. Open the application: `python app.py`
2. Go to "Model Training" tab
3. Set dataset path and parameters
4. Click "Start Training"
5. Monitor progress in the training log

### Using Command Line
1. Update `config.py` with your dataset path
2. Run: `python train_model.py`
3. The trained model will be saved automatically

### Training Parameters
- **Epochs**: Number of training iterations (default: 50)
- **Batch Size**: Training batch size (default: 32)
- **Learning Rate**: Model learning rate (default: 0.001)
- **Data Augmentation**: Enable/disable augmentation

## 🔍 Real-time Monitoring

### Starting Monitoring
1. Ensure a trained model exists
2. Connect your webcam
3. Run the application or monitoring script
4. The system will show live predictions

### Alert System
The system triggers alerts for:
- Sustained disengagement (>30 seconds)
- Low prediction confidence (<50%)
- Custom threshold violations

### Confidence Threshold
Adjust the confidence threshold in settings to control sensitivity:
- **High threshold (0.8-0.9)**: More conservative alerts
- **Low threshold (0.5-0.7)**: More sensitive alerts

## 📈 Analytics and Reporting

### Session Statistics
- Total readings and engagement rate
- Average confidence scores
- Session duration and alert counts
- Real-time engagement charts

### Export Options
- Session reports (JSON format)
- Engagement charts (PNG/PDF)
- Alert logs and statistics

## ⚙️ Configuration

### Key Settings in `config.py`
```python
# Dataset
DATASET_PATH = "path/to/dataset"

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Monitoring
CONFIDENCE_THRESHOLD = 0.7
```

### Application Settings
Configure through the GUI Settings tab:
- Confidence thresholds
- Alert preferences
- Audio notifications

## 🔧 Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **"No trained model found"**
   - Train a model first using the GUI or command line
   - Check that `MODEL_SAVE_PATH` is correctly set

3. **Webcam not detected**
   - Ensure webcam is connected and working
   - Check webcam permissions
   - Try changing camera index in the code

4. **Low accuracy**
   - Increase training epochs
   - Use data augmentation
   - Ensure dataset quality and balance
   - Add more training data

5. **Memory errors during training**
   - Reduce batch size
   - Use data generators instead of loading all data
   - Close other applications

### Performance Optimization

1. **For better accuracy**:
   - Use larger, diverse dataset
   - Enable data augmentation
   - Train for more epochs
   - Fine-tune hyperparameters

2. **For faster inference**:
   - Use GPU acceleration
   - Optimize model architecture
   - Reduce image resolution if needed

## 📋 Model Architecture

The CNN model consists of:
- **Input Layer**: 224x224x3 RGB images
- **Convolutional Layers**: 4 Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for dimensionality reduction
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: Sigmoid activation for binary classification

### Model Summary
```
- Conv2D (32 filters) → MaxPooling2D
- Conv2D (64 filters) → MaxPooling2D  
- Conv2D (128 filters) → MaxPooling2D
- Conv2D (128 filters) → MaxPooling2D
- Flatten → Dense (512) → Dropout
- Dense (1, sigmoid) → Output
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Contributors and dataset providers

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration settings
3. Ensure all dependencies are installed
4. Verify dataset format and paths

## 🔮 Future Enhancements

- Multi-class engagement levels (highly engaged, engaged, neutral, disengaged)
- Facial landmark detection for attention analysis
- Integration with classroom management systems
- Mobile app for remote monitoring
- Advanced analytics with ML insights
- Multi-student tracking in group settings

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Author**: Class Monitoring Team