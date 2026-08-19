# AI Palmistry – Intelligent Palm Analysis & Traditional Interpretation

A complete web application integrating computer vision (OpenCV, MediaPipe) with traditional palmistry lore interpretations.

## Project Structure
```
AI-Palmistry/
├── app.py                      # Flask Application Entry Point
├── config.py                   # App Configuration Settings
├── requirements.txt            # Python Dependencies
├── models/                     # Placeholder directory for ML Models
├── palm_analysis/             # Computer Vision & Interpretation Modules
│   ├── __init__.py
│   ├── preprocessing.py        # Image Denoising & CLAHE Contrast Enhancement
│   ├── hand_detection.py       # MediaPipe 21 Landmark Detection
│   ├── palm_detection.py       # Palm ROI Crop & Alignment
│   ├── line_detection.py       # OpenCV Palm Line Detection (Heart, Head, Life, Fate)
│   ├── feature_extraction.py   # Measurable Feature Metrics Vector Engine
│   └── interpretation.py     # Traditional Lore Rule-based Matrix
├── database/                  # SQLite Database Layer
│   ├── __init__.py
│   └── db.py                   # DB Schemas & Connection Management
├── templates/                  # HTML Templates (Bootstrap 5 + Custom Glassmorphism)
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── analysis.html
│   ├── result.html
│   ├── about.html
│   └── privacy.html
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── images/
└── uploads/, processed/, results/
```

## How to Run

1. Navigate to project folder:
```bash
cd C:\Users\budde\.gemini\antigravity\scratch\AI-Palmistry
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Flask app:
```bash
python app.py
```

4. Open browser at `http://127.0.0.1:5000`.

## Disclaimer
All generated interpretations are based on traditional palmistry lore for entertainment, curiosity, and self-reflection. Not medical, financial, or predictive science advice.
