# 🌿 Plant Disease AI Assistant

A machine learning application that identifies plant diseases from leaf images and provides comprehensive treatment recommendations with multilingual audio support.

## 📋 Features

- **AI Disease Detection**: Uses EfficientNetB0 neural network trained on 15+ plant diseases
- **15 Disease Classifications**:
  - Pepper Bell: Bacterial spot, Healthy
  - Potato: Early blight, Late blight, Healthy
  - Tomato: 11 different diseases and healthy status

- **Comprehensive Treatment Database**:
  - Chemical treatments with dosages
  - Organic alternatives
  - Cultural practices
  - Prevention strategies
  - Product recommendations
  - Action timelines

- **Multilingual Audio Support**: English, Hindi

- **Interactive Web UI**: Built with Streamlit
  - Image upload and preview
  - Real-time prediction with confidence scores
  - Tabbed treatment information
  - Audio playback

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- macOS with Metal GPU support (or Linux/Windows with TensorFlow)

### Installation

1. **Create Conda Environment**:
```bash
conda create -n plant_app python=3.10 -y
conda activate plant_app
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the App**:
```bash
streamlit run app.py
```

4. **Access the App**:
Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
plant-disease-app/
├── app.py                    # Main Streamlit application
├── treatment_module.py       # Treatment database & functions
├── prediction_module.py      # Image prediction pipeline
├── evaluation_module.py      # Model evaluation & metrics
├── train_model.py            # Model training script
├── best_model.keras          # Trained EfficientNetB0 model
├── requirements.txt          # Python dependencies
├── train/                    # Training dataset (15 disease folders)
├── val/                      # Validation dataset
├── test/                     # Test dataset
├── .git/                     # Version control
└── .gitignore               # Git ignore rules
```

### Module Descriptions

- **app.py**: Main Streamlit web application with UI and model integration
- **treatment_module.py**: Comprehensive treatment database with 15 plant diseases
- **prediction_module.py**: Image preprocessing and disease prediction pipeline
- **evaluation_module.py**: Model evaluation, metrics, and visualization tools
- **train_model.py**: Two-phase model training with frozen base and fine-tuning

## 💾 Requirements

See `requirements.txt`:
- streamlit >= 1.28.0
- tensorflow-macos == 2.10.0 (macOS with Metal)
- numpy >= 1.23.0
- pillow >= 10.0.0
- scipy >= 1.11.0
- gtts >= 2.4.0

## 🎯 How to Use

### Streamlit Web App

1. **Upload a Leaf Image**: Click "Upload Leaf Image" and select a JPG, JPEG, or PNG file
2. **View Prediction**: See the disease prediction with confidence percentage
3. **Read Treatment Advice**: Browse 5 tabs of detailed information
   - Quick Summary
   - Chemical Treatments
   - Organic Treatments
   - Cultural Practices
   - Prevention Tips
4. **Listen to Audio**: Click "Play Treatment Audio" in your selected language

### Prediction Module (Programmatic)

Use `prediction_module.py` for batch predictions:

```python
from prediction_module import PlantDiseasePredictor, predict_single_image

# Initialize predictor
predictor = PlantDiseasePredictor()

# Predict single image
result = predict_single_image("path/to/leaf.jpg")
print(result)  # {'disease': 'Tomato_Early_blight', 'confidence': 0.92, 'predictions': [...]}

# Or use class directly
predictor = PlantDiseasePredictor()
predictions = predictor.predict("path/to/leaf.jpg")  # Top-1 prediction
predictions = predictor.predict("path/to/leaf.jpg", top_n=5)  # Top-5 predictions
```

### Evaluation Module (Model Metrics)

Evaluate model performance on validation dataset:

```python
from evaluation_module import full_evaluation

# Generate comprehensive evaluation report
report = full_evaluation(
    dataset_path="val/",
    model_path="best_model.keras",
    batch_size=32
)
# Outputs: confusion matrix, classification report, per-class metrics, visualizations
```

### Training Model

Retrain the model with two-phase training:

```bash
python train_model.py
```

This will:
1. Train for 5 epochs with frozen base model (fast training)
2. Fine-tune for 5 epochs with learnable base (improved accuracy)
3. Save checkpoints after each phase
4. Display training/validation metrics

## 🔍 Treatment Information Includes

For each disease:
- **Severity Level** (None to Very High)
- **Disease Cause** & Symptoms
- **Multiple Treatment Options**:
  - Specific fungicides/bactericides with dosages
  - Organic alternatives
  - Non-chemical management practices
- **Prevention Strategies**
- **Timeline for Treatment Response**
- **Recommended Products** with costs and effectiveness

## 🧠 Model Details

- **Architecture**: EfficientNetB0 with transfer learning
- **Base Model**: ImageNet pre-trained weights
- **Input Size**: 160x160 pixels
- **Classes**: 15 plant disease categories
- **Data Augmentation**: Random flips, rotations, zoom, contrast adjustments

## 📊 Accuracy

- Training Accuracy: ~87%
- Validation Accuracy: ~85%
- Well-balanced across all disease classes

## 🛠 Customization

### Add New Diseases

Edit `treatment_module.py` and add to `TREATMENT_DATABASE`:
```python
"Your_Disease_Name": {
    "severity": "Medium",
    "cause": "...",
    "symptoms": "...",
    "treatments": { ... },
    "prevention": [ ... ]
}
```

### Retrain Model

Create a new training script or modify the data loading in `app.py`.

## 📝 Languages Supported

- English (en)
- Hindi (hi)
- Marathi (mr)

Add more by modifying the language selector in `app.py`.

## ⚠️ Disclaimer

This is an AI-based suggestion system. Always consult agricultural experts for final treatment decisions.

## 📄 License

Project created for agricultural education and disease management assistance.

## 👨‍💻 Development

Built with:
- TensorFlow/Keras for deep learning
- Streamlit for web interface
- gTTS for text-to-speech
- EfficientNetB0 for image classification

---

**Plant Disease AI Assistant** - Helping farmers identify and treat crop diseases with AI 🌱
