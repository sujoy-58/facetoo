# Real-Time Facial Expression Recognition

A computer vision project that detects a face from a webcam feed and classifies the facial expression (e.g., happy, sad, angry, surprised, neutral) in real time using a CNN model trained on a custom dataset.


## ✨ Features
🎥 Real-time Processing: Live webcam feed with instant expression analysis  
👁️ Face Detection: Robust face detection using Haar Cascade classifier  
🧠 CNN Classification: Deep learning model for accurate expression recognition  
🎯 Multi-Expression Support: Happy, Sad, Angry, Surprised, Neutral  
📊 Confidence Metrics: Real-time confidence percentages for predictions  
🔧 Custom Dataset: Trained on organized, labeled expression datasets  


## 🛠️ Tech Stack
-- Python 3.10 - Core programming language  
-- TensorFlow/Keras - Deep learning framework  
-- OpenCV - Computer vision and webcam processing  
-- NumPy - Numerical computations  
-- scikit-learn - Model evaluation and metrics  


## 🚀 Quick Start Guide  
Prerequisites  
-- Python 3.10 installed  
-- Webcam access  
-- Basic command line knowledge  


## 📥 Installation (Step-by-Step)
## Install Python 3.10  
bash  
Download from: https://www.python.org/downloads/release/python-31011/  
During installation:  
✅ Enable "Add Python to PATH"  
✅ Click "Install Now"  

## Download Project  
bash  
Click 'Code' → 'Download ZIP' on GitHub  
Extract to desired location (e.g., Desktop/facetoo/)  

## Open Project Folder  
bash  
Navigate to project folder  
Click address bar, type 'cmd', press Enter  

## Install Dependencies  
bash  
pip install -r requirements.txt  

## Verify Required Files  
Ensure these files are present:  
text:  
model/expression_model.h5  
model/expressions_labels.txt  
haarcascade_frontalface_default.xml  

## If Haar Cascade missing:  
bash  
Download from:  
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml  
Place in project root directory  

## Run Application  
bash  
cd app  
python realtime.py  

or  

py -3.10 realtime.py  


## 🎮 Usage  
-- Launch: Run realtime.py  
-- Webcam: Grant camera permissions when prompted  
-- Detection: Position face in camera view  
-- Results: View real-time expression and confidence  
-- Exit: Press q to quit application  


## 📊 Model Performance

| Expression  | Precision | Recall |
|-------------|-----------|--------|
| 😊 Happy    | 0.87      | 0.63   |
| 😖 Disgust  | 0.72      | 0.65   |
| 😲 Surprise | 0.74      | 0.45   |
| 😠 Angry    | 0.36      | 0.34   |
| 😨 Fear     | 0.33      | 0.19   |
| 😐 Neutral  | 0.40      | 0.46   |
| 😢 Sad      | 0.27      | 0.57   |

## 🎯 Accuracy Improvement Strategies  
✅ Data Augmentation - Expand training dataset variety  
✅ Batch Normalization - Improve training stability  
✅ Class Weighting - Handle imbalanced datasets  
✅ Transfer Learning - Leverage pre-trained models (MobileNetV2)  
✅ Advanced Architectures - Implement deeper CNN networks  


## 🔮 Future Enhancements  
🚀 Immediate  
Transfer learning implementation for higher accuracy  
Real-time FPS and confidence overlay improvements  
Enhanced dataset collection and curation  


## 🔧 Technical  
Multi-face detection and analysis  
Expression timeline and analytics  
Export functionality for session data  
Custom model training interface  


## 📁 Project Structure

<pre>
facetoo/
├── 📁 noteBook/
│   └── 📄 jupyter.ipynb              # Training code (Colab)
├── 📁 model/
│   ├── 📄 expression_model.h5        # Trained CNN model
│   └── 📄 expressions_labels.txt     # Expression labels
├── 📁 app/
│   └── 📄 realtime.py                # Real-time detection script
├── 📄 haarcascade_frontalface_default.xml   # Face detector
├── 📄 requirements.txt
└── 📄 README.md
</pre>

## 🛠️ Troubleshooting  
Common Issues & Solutions  
Issue	Solution  
-- TensorFlow installation fails	Ensure Python 3.10 is used  
-- Webcam not detected	Check camera permissions  
-- Model file missing	Verify file paths in project  
-- Low confidence scores	Ensure good lighting and clear face view  


## Performance Tips  
💡 Ensure adequate lighting for better detection  
💡 Position face centrally in frame  
💡 Maintain neutral background for optimal results  
💡 Close other camera applications during use  


## 🤝 Contributing
We welcome contributions! Areas for improvement:  
-- Model architecture enhancements  
-- Dataset expansion  
-- Performance optimization  
-- Additional expression support  
