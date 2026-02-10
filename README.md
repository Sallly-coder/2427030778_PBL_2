# 🎙️ Speech Emotion Recognition (SER) — PBL Project (Sem 2)

> **Project-Based Learning | 6-Semester Project | Semester 2 Mid-Term**

---

## 📌 Overview

This project implements a **Speech Emotion Recognition (SER)** system that classifies human emotions from audio signals using:

- Audio signal processing (`librosa`)
- Feature extraction (MFCCs, pitch, energy)
- Machine learning classifiers (Logistic Regression, Random Forest, MLP)

---

## 🗂️ Project Structure

```
speech-emotion-recognition-pbl2/
├── data/
│   └── README.md               # Dataset info & download instructions
├── notebooks/
│   ├── 01_signal_processing_demo.ipynb   # Audio loading & visualization
│   └── 02_feature_extraction.ipynb       # MFCC extraction & EDA
├── src/
│   ├── audio_processing.py     # Load, normalize, denoise audio
│   ├── feature_extraction.py   # MFCC, pitch, energy features
│   ├── model.py                # Classifier definitions
│   └── train_eval.py           # Training loop & evaluation metrics
├── docs/
│   └── mid_term_presentation.html  # GitHub Pages site
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_ID/speech-emotion-recognition-pbl2.git
cd speech-emotion-recognition-pbl2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
See `data/README.md` for instructions (RAVDESS / TESS).

### 4. Run training
```bash
python src/train_eval.py --data_dir data/ravdess_subset --emotions happy sad neutral
```

### 5. Explore notebooks
```bash
jupyter notebook notebooks/
```

---

## 🎯 Emotions Targeted (Mid-Term)

| Label     | Code |
|-----------|------|
| Neutral   | 01   |
| Happy     | 03   |
| Sad       | 04   |
| Angry     | 05   |

*(RAVDESS emotion codes)*

---

## 📊 Mid-Term Results (Baseline)

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~65–70%  |
| Random Forest       | ~70–75%  |
| MLP Classifier      | ~72–78%  |

> *Results on 3-emotion subset (happy/sad/neutral), MFCC mean+std features*

---

## 🛣️ Roadmap

- [x] **Sem 2 (Mid-Term):** Pipeline setup, MFCC extraction, baseline classifiers
- [ ] **Sem 3:** 1D-CNN / LSTM models, more emotions
- [ ] **Sem 4:** Real-time audio input, Flask web app
- [ ] **Sem 5–6:** Deployment, UI polish, paper/report

---

## 🔗 Links

- **GitHub Pages / Presentation:** [Link here]
- **Dataset:** [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

---

## 📚 References

- [Librosa Documentation](https://librosa.org/doc/)
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [TechVidvan – Speech Emotion Recognition](https://techvidvan.com/tutorials/python-project-speech-emotion-recognition/)
- Scikit-learn User Guide: https://scikit-learn.org/stable/
