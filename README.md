# 🎤 AI Presentation Coach

An advanced AI-based system designed to evaluate and improve presentation skills using real-time analysis of facial expressions, body posture, and speech patterns.

This project combines **Computer Vision**, **Speech Processing**, and **Machine Learning** to provide intelligent feedback and confidence scoring for public speaking practice.

---

## 🚀 Key Features

### 🎥 Visual Analysis

* Real-time face detection using MediaPipe
* Eye contact tracking and scoring
* Head tilt and movement analysis
* Posture evaluation (shoulder alignment, body position)
* Facial landmark tracking

### 🎤 Audio & Speech Analysis

* Real-time microphone input processing
* Voice activity detection (speaking vs silence)
* Volume and energy analysis
* Speech pace detection (slow / normal / fast)
* Silence ratio calculation

### 🧠 AI & Machine Learning

* Trained ML model for presentation evaluation
* Feature-based prediction system
* Confidence scoring (0–100)
* Real-time performance tracking

### 📊 Feedback System

* Live feedback generation (good / warning / improvement)
* Personalized suggestions
* Performance trend tracking
* Confidence analytics (momentum, consistency)

### 💾 Data Handling

* Session logging (CSV export)
* Feature extraction pipeline
* Dataset-based model training

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit** – UI & interaction
* **OpenCV** – Computer vision processing
* **MediaPipe** – Face & pose detection
* **NumPy / Pandas** – Data processing
* **Scikit-learn** – Machine learning model
* **Matplotlib / Plotly** – Visualization
* **SoundDevice / SpeechRecognition** – Audio processing

---

## 📂 Project Structure

```
AI-Presentation-Coach/
│
├── app.py                  # Main Streamlit application
├── engine.py               # Core AI processing engine
├── step1_collect_data.py   # Feature extraction module
├── model.pkl               # Trained ML model
├── dataset.csv             # Training dataset
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (ffmpeg)
├── .gitignore
└── README.md
```

---

## ▶️ How to Run (Local Setup)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-presentation-coach.git
cd ai-presentation-coach
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## ⚠️ System Requirements

* Webcam (for face & posture detection)
* Microphone (for speech analysis)
* Python 3.9+
* Recommended: Good lighting for accurate detection

---

## 🌐 Deployment Note

This project is designed primarily for **local execution** because it uses:

* Real-time **microphone input**
* Real-time **camera access**

⚠️ Cloud platforms (like Streamlit Cloud) have limitations:

* Microphone access is not supported
* Camera support is limited

---

## 🎯 Use Cases

* Presentation practice & improvement
* Public speaking training
* Interview preparation
* Confidence building
* AI-based behavioral analysis

---

## 📈 Future Enhancements

* Cloud-compatible audio processing
* Advanced NLP-based speech feedback
* Emotion detection
* Dashboard with detailed analytics
* Multi-user support

---

## 👨‍💻 Author

**Heet Jain**
B.Tech (Information Technology)


---
