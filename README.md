# Animal Type Classification (ATC) System

A prototype system for classifying cows and buffaloes based on body structure traits using computer vision and machine learning.

## Overview

This system uses image processing to extract measurable body structure features from animal photos and classifies them into productivity categories (High/Medium/Low) based on traits like:

- Body length and height
- Chest width 
- Rump angle and hip width
- Udder depth (for females)
- Leg angle and foot structure

## System Architecture

```
ATC/
├── backend/           # FastAPI backend
│   ├── main.py       # API endpoints
│   ├── models/       # ML models and data models
│   └── utils/        # Image processing utilities
├── frontend/         # Streamlit web interface
│   └── app.py
├── data/             # Training data and models
├── uploads/          # Temporary image storage
└── requirements.txt
```

## Setup Instructions

### 1. Clone and Setup Environment

```bash
cd C:\Users\pc\OneDrive\Desktop\ATC
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Backend Server

```bash
cd backend
python main.py
```

Backend will run on: http://localhost:8000

### 3. Start Frontend (New Terminal)

```bash
cd frontend
streamlit run app.py
```

Frontend will run on: http://localhost:8501

## Features

- **Image Upload**: Upload cow/buffalo images for analysis
- **Species Selection**: Choose between cow or buffalo
- **Feature Extraction**: Automatic extraction of body measurements
- **Classification**: AI-powered productivity assessment
- **Scorecard**: Detailed trait-wise scoring
- **History**: Optional storage of classification results

## API Endpoints

- `POST /upload`: Upload and process animal image
- `GET /health`: Health check endpoint
- `GET /history`: Get classification history

## Usage

1. Open the Streamlit frontend at http://localhost:8501
2. Upload an image of a cow or buffalo
3. Select the animal species
4. View extracted measurements and productivity classification
5. Review detailed trait scorecard

## Technology Stack

- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **Image Processing**: OpenCV
- **Machine Learning**: scikit-learn (Random Forest)
- **Database**: SQLite
- **Computer Vision**: OpenCV, NumPy

## Model Details

The classification model considers:

1. **Body Structure Metrics**:
   - Body length to height ratio
   - Chest circumference indicators
   - Leg positioning and angles

2. **Productivity Indicators**:
   - High: Optimal body proportions for milk/meat production
   - Medium: Good structure with minor limitations
   - Low: Structural issues affecting productivity

## Future Enhancements

- Deep learning CNN models
- More precise feature extraction algorithms
- Breed-specific classification
- Real-time video processing
- Mobile app integration

## License

MIT License - See LICENSE file for details